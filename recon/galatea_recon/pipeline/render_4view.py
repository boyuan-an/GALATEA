import os
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import argparse
import pickle
import numpy as np
import trimesh
import pyrender
import imageio
from PIL import Image, ImageDraw, ImageFont

HAND_COLOR = [225, 110, 140, 255]
OBJ_COLOR = [110, 195, 255, 255]


def load_mano_faces(args):
    if args.mano_faces:
        return np.load(args.mano_faces).astype(np.int32)
    with open(args.mano_pkl, 'rb') as f:
        m = pickle.load(f, encoding='latin1')
    return np.asarray(m['f']).astype(np.int32)


def load_K(cam_dir):
    fx, fy, cx, cy, W, H = np.loadtxt(os.path.join(cam_dir, 'cam_intrinsics.txt')).tolist()
    return fx, fy, int(W), int(H)


def look_at(eye, target, up=(0, 0, 1)):
    eye = np.asarray(eye, float); target = np.asarray(target, float); up = np.asarray(up, float)
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    M = np.eye(4)
    M[:3, 0] = r; M[:3, 1] = u; M[:3, 2] = -f; M[:3, 3] = eye
    return M


def cv_to_gl(T_cv):
    return T_cv @ np.diag([1, -1, -1, 1]).astype(float)


def add_label(img, text):
    pil = Image.fromarray(img); d = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 18)
    except Exception:
        font = ImageFont.load_default()
    d.rectangle([(0, 0), (max(140, 8 * len(text)), 28)], fill=(0, 0, 0))
    d.text((6, 4), text, fill=(255, 255, 255), font=font)
    return np.asarray(pil)


def build_scene(hand_v, hand_f, obj_v, obj_f):
    scene = pyrender.Scene(bg_color=[0.93, 0.93, 0.95, 1.0], ambient_light=[0.45, 0.45, 0.45])
    hm = trimesh.Trimesh(vertices=hand_v, faces=hand_f, process=False)
    hm.visual.vertex_colors = HAND_COLOR
    scene.add(pyrender.Mesh.from_trimesh(hm, smooth=True))
    om = trimesh.Trimesh(vertices=obj_v, faces=obj_f, process=False)
    om.visual.vertex_colors = OBJ_COLOR
    scene.add(pyrender.Mesh.from_trimesh(om, smooth=True))
    for eye in [(1.0, 1.0, 1.8), (-1.2, 0.8, 1.2), (0.0, -1.5, 1.0)]:
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.5),
                  pose=look_at(list(eye), [0, 0, 0.1]))
    return scene


def render_view(renderer, scene, cam_pose_gl, yfov_deg):
    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(yfov_deg), znear=0.01, zfar=20.0)
    node = scene.add(cam, pose=cam_pose_gl)
    color, _ = renderer.render(scene)
    scene.remove_node(node)
    return color


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--obj_poses', required=True)
    ap.add_argument('--hand_verts', required=True)
    ap.add_argument('--obj_mesh', required=True)
    ap.add_argument('--cam_dir', required=True, help='H2O cam0 dir (for the real-camera-view fov)')
    ap.add_argument('--mano_faces', default=None, help='mano_right_faces.npy (preferred)')
    ap.add_argument('--mano_pkl', default=None, help='MANO_RIGHT.pkl (fallback, reads "f")')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--fps', type=int, default=15)
    ap.add_argument('--keyframe_every', type=int, default=10)
    ap.add_argument('--view_size', type=int, default=480)
    ap.add_argument('--max_frames', type=int, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    if not args.mano_faces and not args.mano_pkl:
        cand = os.path.join(os.path.dirname(os.path.expanduser(args.hand_verts)), 'mano_right_faces.npy')
        if os.path.exists(cand):
            args.mano_faces = cand
    assert args.mano_faces or args.mano_pkl, 'need --mano_faces or --mano_pkl (or mano_right_faces.npy beside hand_verts)'
    os.makedirs(args.out_dir, exist_ok=True)

    obj_poses = np.load(os.path.expanduser(args.obj_poses)).astype(np.float64)
    hverts = np.load(os.path.expanduser(args.hand_verts)).astype(np.float64)
    hand_f = load_mano_faces(args)
    mesh = trimesh.load(os.path.expanduser(args.obj_mesh), force='mesh')
    Vobj = np.asarray(mesh.vertices, np.float64)
    obj_f = np.asarray(mesh.faces, np.int32)
    fx, fy, W, H = load_K(args.cam_dir)

    N = min(len(obj_poses), len(hverts))
    if args.max_frames:
        N = min(N, args.max_frames)

    obj_v_all = [Vobj @ obj_poses[i][:3, :3].T + obj_poses[i][:3, 3] for i in range(N)]
    pts = np.concatenate([hverts[:N].reshape(-1, 3)[::8]] +
                         [ov[::20] for ov in obj_v_all], 0)
    center = 0.5 * (pts.min(0) + pts.max(0))
    radius = float(np.linalg.norm(pts - center, axis=1).max())
    r = max(0.18, radius)
    RIGHT, UP, FWD = np.array([1., 0, 0]), np.array([0, -1., 0]), np.array([0, 0, 1.])

    cam_follow = cv_to_gl(np.eye(4))
    yfov_cam = float(np.rad2deg(2 * np.arctan((H / 2) / fy)))
    eye_persp = center + r * (0.85 * RIGHT + 0.85 * UP - 1.1 * FWD)
    eye_top = center + r * 1.9 * UP
    eye_side = center + r * 1.7 * RIGHT
    views_static = [
        (look_at(eye_persp, center, up=UP), 'Perspective', 50.0),
        (look_at(eye_top, center, up=FWD), 'Top', 50.0),
        (look_at(eye_side, center, up=UP), 'Side', 50.0),
    ]

    renderer = pyrender.OffscreenRenderer(args.view_size, args.view_size)
    mp4 = os.path.join(args.out_dir, 'solid_4view.mp4')
    writer = imageio.get_writer(mp4, fps=args.fps, macro_block_size=1, codec='libx264', quality=8)

    for i in range(N):
        scene = build_scene(hverts[i], hand_f, obj_v_all[i], obj_f)
        imgs = [add_label(render_view(renderer, scene, cam_follow, yfov_cam),
                          f'Camera (f{i + 1:03d})')]
        for T, label, yf in views_static:
            imgs.append(add_label(render_view(renderer, scene, T, yf), label))
        grid = np.concatenate([np.concatenate([imgs[0], imgs[1]], 1),
                               np.concatenate([imgs[2], imgs[3]], 1)], 0)
        writer.append_data(grid)
        if i % args.keyframe_every == 0:
            imageio.imwrite(os.path.join(args.out_dir, f'keyframe_{i:04d}.png'), grid)
        if i % 10 == 0 or i == N - 1:
            print(f'  rendered {i + 1}/{N}')

    writer.close()
    renderer.delete()
    print(f'mp4: {mp4}')


if __name__ == '__main__':
    main()
