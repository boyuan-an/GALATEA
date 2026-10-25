import numpy as np

import arm_ik


def solve_continuous(targets, bT, rng, q0, hard=0.01, rot_hard=np.radians(5.0), repair_cap=0.295, repair_passes=14):
    armq, n_unreach = arm_ik.solve_branches(
        targets, bT, rng, q0, n_restart=12, cont_it=200, restart_it=160, hard=hard, rot_hard=rot_hard, beam=10
    )
    n = len(targets)

    def try_repair(t, anchor_q):
        q, ep, er = arm_ik.solve_ball(targets[t], bT, anchor_q, repair_cap)
        if ep <= hard and er <= rot_hard:
            return q
        q2, ep2, er2 = arm_ik.solve_restarts(targets[t], bT, anchor_q, rng)
        if ep2 <= hard and er2 <= rot_hard:
            return q2
        return None

    for _ in range(repair_passes):
        changed = False
        for t in range(1, n):
            d = np.linalg.norm(armq[t] - armq[t - 1])
            if d > repair_cap:
                q = try_repair(t, armq[t - 1])
                if q is not None and np.linalg.norm(q - armq[t - 1]) < d - 1e-06:
                    armq[t] = q
                    changed = True
        for t in range(n - 2, 0, -1):
            d = np.linalg.norm(armq[t] - armq[t + 1])
            if d > repair_cap:
                q = try_repair(t, armq[t + 1])
                if q is not None and np.linalg.norm(q - armq[t + 1]) < d - 1e-06:
                    armq[t] = q
                    changed = True
        if not changed:
            break
    if not np.array_equal(armq[0], np.asarray(q0, dtype=np.float64)):
        raise RuntimeError("continuity repair changed the frame-0 configuration")
    pos_err = np.empty(n)
    rot_err = np.empty(n)
    for t in range(n):
        pos_err[t], rot_err[t] = arm_ik.pose_error(armq[t], bT, targets[t])
    return (armq, pos_err, rot_err, n_unreach)
