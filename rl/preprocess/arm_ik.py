import xml.etree.ElementTree as ET
from math import cos, sin

import numpy as np
from scipy.spatial.transform import Rotation as R

ARM_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
TARGET_LINK = "right_hand_C_MC"
POS_TOL, ROT_TOL = (0.0002, 0.002)
MAX_IT = 600
RESTART_IT = 300
N_RESTART = 80
GATE = 0.002
HARD = 0.01
ROT_HARD = np.radians(5.0)
K_NULL = 0.08
DQ_CLAMP = 0.12
DAMP = 0.0016
SPEED_CAP = 0.6

CHAIN = LO = HI = None
_I7, _I6 = (np.eye(7), np.eye(6))


def T_from(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz
    return T


def aa_R(ax, ang):
    a = np.asarray(ax, float)
    a = a / np.linalg.norm(a)
    c, s = (cos(ang), sin(ang))
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def logSO3(Rm):
    return R.from_matrix(Rm).as_rotvec()


def configure(urdf):
    global CHAIN, LO, HI
    root = ET.parse(urdf).getroot()
    pmap = {}
    for j in root.findall("joint"):
        p = j.find("parent").get("link")
        c = j.find("child").get("link")
        o = j.find("origin")
        xyz, rpy = ([0, 0, 0], [0, 0, 0])
        if o is not None:
            if o.get("xyz"):
                xyz = [float(x) for x in o.get("xyz").split()]
            if o.get("rpy"):
                rpy = [float(x) for x in o.get("rpy").split()]
        ax = [0, 0, 1]
        a = j.find("axis")
        if a is not None and a.get("xyz"):
            ax = [float(x) for x in a.get("xyz").split()]
        lim = j.find("limit")
        lo, hi = (-3.14, 3.14)
        if lim is not None:
            if lim.get("lower"):
                lo = float(lim.get("lower"))
            if lim.get("upper"):
                hi = float(lim.get("upper"))
        pmap[c] = (p, T_from(xyz, rpy), j.get("type"), np.array(ax, float), j.get("name"), lo, hi)
    chain, cur = ([], TARGET_LINK)
    while cur in pmap:
        chain.append(pmap[cur])
        cur = pmap[cur][0]
    CHAIN = chain[::-1]
    LO = np.array([[pmap[k][5] for k in pmap if pmap[k][4] == jn][0] for jn in ARM_JOINTS])
    HI = np.array([[pmap[k][6] for k in pmap if pmap[k][4] == jn][0] for jn in ARM_JOINTS])


def base_transform(base_pos, base_yaw):
    bT = np.eye(4)
    bT[:3, :3] = aa_R([0, 0, 1], base_yaw)
    bT[:3, 3] = base_pos
    return bT


def target_pose(pos, rot_aa):
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(rot_aa).as_matrix()
    T[:3, 3] = pos
    return T


def fk(q, bT):
    T = bT.copy()
    js = []
    for p, To, jt, ax, nm, lo, hi in CHAIN:
        T = T @ To
        if jt == "revolute" and nm in ARM_JOINTS:
            js.append((T[:3, :3] @ ax, T[:3, 3].copy()))
            Rq = np.eye(4)
            Rq[:3, :3] = aa_R(ax, q[ARM_JOINTS.index(nm)])
            T = T @ Rq
    return (T, js)


def pose_error(q, bT, T):
    current, _ = fk(q, bT)
    return (
        np.linalg.norm(T[:3, 3] - current[:3, 3]),
        np.linalg.norm(logSO3(T[:3, :3] @ current[:3, :3].T)),
    )


def _jacobian_step(Tt, T, js, damp):
    ep = Tt[:3, 3] - T[:3, 3]
    er = logSO3(Tt[:3, :3] @ T[:3, :3].T)
    if np.linalg.norm(ep) < POS_TOL and np.linalg.norm(er) < ROT_TOL:
        return None
    pe = T[:3, 3]
    J = np.zeros((6, 7))
    for i, (zw, pw) in enumerate(js):
        J[:3, i] = np.cross(zw, pe - pw)
        J[3:, i] = zw
    Jpinv = J.T @ np.linalg.solve(J @ J.T + damp * _I6, _I6)
    return (J, Jpinv, Jpinv @ np.concatenate([ep, er]))


def solve(Tt, bT, seed, qref, max_it, damp=DAMP):
    q = seed.copy()
    for _ in range(max_it):
        T, js = fk(q, bT)
        step = _jacobian_step(Tt, T, js, damp)
        if step is None:
            break
        J, Jpinv, dq = step
        if qref is not None:
            dq = dq + (_I7 - Jpinv @ J) @ (K_NULL * (qref - q))
        q = np.clip(q + np.clip(dq, -DQ_CLAMP, DQ_CLAMP), LO, HI)
    return (q, *pose_error(q, bT, Tt))


def solve_ball(Tt, bT, anchor, radius, max_it=MAX_IT):
    q = anchor.copy()
    for _ in range(max_it):
        T, js = fk(q, bT)
        step = _jacobian_step(Tt, T, js, DAMP)
        if step is None:
            break
        q = q + np.clip(step[2], -DQ_CLAMP, DQ_CLAMP)
        d = q - anchor
        n = np.linalg.norm(d)
        if n > radius:
            q = anchor + d * (radius / n)
        q = np.clip(q, LO, HI)
    return (q, *pose_error(q, bT, Tt))


def solve_restarts(Tt, bT, qprev, rng):
    q, ep, er = solve(Tt, bT, qprev, qprev, MAX_IT)
    if ep > GATE or er > ROT_HARD:
        q2, ep2, er2 = solve(Tt, bT, qprev, qprev, MAX_IT * 2, damp=DAMP * 0.2)
        if ep2 < ep:
            q, ep, er = (q2, ep2, er2)
    if ep <= HARD and er <= ROT_HARD:
        return (q, ep, er)
    cands = [(q, ep, er)]
    n_good = 0
    for _ in range(N_RESTART):
        q2, ep2, er2 = solve(Tt, bT, rng.uniform(LO, HI), None, RESTART_IT)
        cands.append((q2, ep2, er2))
        n_good += int(ep2 < HARD and er2 < ROT_HARD)
        if n_good >= 8:
            break
    good = [c for c in cands if c[1] < HARD and c[2] < ROT_HARD]
    if good:
        return min(good, key=lambda c: np.linalg.norm(c[0] - qprev))
    pos_ok = [c for c in cands if c[1] < HARD]
    if pos_ok:
        return min(pos_ok, key=lambda c: c[2])
    return min(cands, key=lambda c: c[1])


def _dedup(cands, tol=0.3):
    keep = []
    for q in cands:
        if all((np.linalg.norm(q - k) > tol for k in keep)):
            keep.append(q)
    return keep


def solve_branches(targ, bT, rng, q0, n_restart, cont_it, restart_it, hard, rot_hard, beam):
    q0 = np.asarray(q0, float)
    seed_pos_error, seed_rot_error = pose_error(q0, bT, targ[0])
    if seed_pos_error >= hard or seed_rot_error >= rot_hard:
        raise RuntimeError(
            f"frame-0 seed violates the IK gate: pos={seed_pos_error * 1000.0:.3f}mm rot={np.degrees(seed_rot_error):.3f}deg"
        )
    N = len(targ)
    cand = [[q0.copy()]]
    n_unreach = 0
    for t in range(1, N):
        cs = []
        for s in cand[-1]:
            q, ep, er = solve(targ[t], bT, s, None, cont_it)
            if ep < hard and er < rot_hard:
                cs.append(q)
        cs = _dedup(cs)
        for _ in range(n_restart if len(cs) < 2 else 2):
            q2, ep2, er2 = solve(targ[t], bT, rng.uniform(LO, HI), None, restart_it)
            if ep2 < hard and er2 < rot_hard:
                cs.append(q2)
        if cs:
            cs = _dedup(cs)[:beam]
        else:
            q, _, _ = solve_ball(targ[t], bT, cand[-1][0], SPEED_CAP)
            cs = [q]
            n_unreach += 1
        cand.append(cs)
    cost = [0.0]
    ptr = [[-1]]
    for t in range(1, N):
        newcost = [0.0] * len(cand[t])
        newptr = [0] * len(cand[t])
        for j, qj in enumerate(cand[t]):
            best = 1e18
            bi = 0
            for i, qi in enumerate(cand[t - 1]):
                c = cost[i] + float(np.linalg.norm(qj - qi))
                if c < best:
                    best = c
                    bi = i
            newcost[j] = best
            newptr[j] = bi
        cost = newcost
        ptr.append(newptr)
    armq = np.zeros((N, 7))
    j = int(np.argmin(cost))
    for t in range(N - 1, 0, -1):
        armq[t] = cand[t][j]
        j = ptr[t][j]
    armq[0] = cand[0][j]
    return (armq, n_unreach)
