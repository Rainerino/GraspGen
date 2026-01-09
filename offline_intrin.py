# -*- coding: utf-8 -*-
import os
import json
import glob
import numpy as np
import cv2
import open3d as o3d


# ============================================================
# 1) mask -> depth 尺寸：支持“RGB黑底mask”（背景黑，物体保留RGB）
# ============================================================
def warp_mask_to_depth(
    mask_img: np.ndarray,
    out_hw: tuple[int, int],
    *,
    mode: str = "rgb_nonblack",
    nonblack_thresh: int = 10,
    bin_thresh: int = 0,
    morph_close: int = 0,
    morph_open: int = 0,
) -> np.ndarray:
    """
    将 mask 对齐到 depth 尺寸(out_hw=(H,W))，支持：
      - rgb_nonblack：你的mask是RGB图，背景纯黑(0,0,0)，物体区域非黑 -> 前景
      - gray：普通灰度/二值mask，>bin_thresh -> 前景
      - auto：自动判断（这里默认你就是 rgb_nonblack）

    返回: (H_out, W_out) bool
    """
    if mask_img is None:
        raise ValueError("mask_img is None")

    H_out, W_out = out_hw

    # 只做一次 resize 到 depth 尺寸（最近邻避免边界模糊）
    if mask_img.shape[:2] != (H_out, W_out):
        mask_rs = cv2.resize(mask_img, (W_out, H_out), interpolation=cv2.INTER_NEAREST)
    else:
        mask_rs = mask_img

    if mode == "auto":
        # 有3通道优先按非黑，否则走灰度
        if mask_rs.ndim == 3 and mask_rs.shape[2] >= 3:
            mode = "rgb_nonblack"
        else:
            mode = "gray"

    if mode == "rgb_nonblack":
        if mask_rs.ndim != 3 or mask_rs.shape[2] < 3:
            raise ValueError(f"rgb_nonblack expects 3-channel mask, got {mask_rs.shape}")

        bgr = mask_rs[:, :, :3].astype(np.int32)
        s = bgr[:, :, 0] + bgr[:, :, 1] + bgr[:, :, 2]
        mb = (s > int(nonblack_thresh))

    elif mode == "gray":
        if mask_rs.ndim == 3:
            mg = cv2.cvtColor(mask_rs[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            mg = mask_rs

        # 处理 16bit
        if mg.dtype == np.uint16:
            mg8 = (mg / 257).astype(np.uint8)
        else:
            mg8 = mg.astype(np.uint8)

        if bin_thresh <= 0:
            mb = (mg8 > 0)
        else:
            mb = (mg8 > int(bin_thresh))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 可选形态学清理
    mb_u8 = (mb.astype(np.uint8) * 255)
    if morph_close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
        mb_u8 = cv2.morphologyEx(mb_u8, cv2.MORPH_CLOSE, k)
    if morph_open > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open, morph_open))
        mb_u8 = cv2.morphologyEx(mb_u8, cv2.MORPH_OPEN, k)

    return (mb_u8 > 0)


def maybe_invert_mask(mask: np.ndarray, mode: str = "keep") -> np.ndarray:
    """
    mode:
      - "auto": True 占比过大(>0.5)认为前景/背景反了 -> 取反
      - "invert": 强制取反
      - "keep": 不取反（建议你先用 keep）
    """
    if mode == "invert":
        return ~mask
    if mode == "keep":
        return mask
    if float(mask.mean()) > 0.5:
        return ~mask
    return mask


# ============================================================
# 2) depth + K + w2c -> 世界点（同时返回 uv 以便从图像采色）
# ============================================================
def backproject_points_with_uv(depth: np.ndarray, K: np.ndarray, w2c_3x4: np.ndarray, mask: np.ndarray):
    if depth.ndim != 2:
        raise ValueError(f"depth must be 2D (H,W), got {depth.shape}")

    H, W = depth.shape
    if mask.shape != (H, W):
        raise ValueError(f"mask shape {mask.shape} != depth shape {depth.shape}")

    valid = mask & np.isfinite(depth) & (depth > 1e-6)
    v, u = np.nonzero(valid)
    if len(u) == 0:
        return np.zeros((0, 3), dtype=np.float32), (u, v)

    z = depth[v, u].astype(np.float32)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    x = (u.astype(np.float32) - cx) * z / fx
    y = (v.astype(np.float32) - cy) * z / fy
    cam = np.stack([x, y, z], axis=1)

    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :4] = w2c_3x4.astype(np.float32)
    c2w = np.linalg.inv(w2c)

    cam_h = np.concatenate([cam, np.ones((cam.shape[0], 1), np.float32)], axis=1)
    world = (c2w @ cam_h.T).T[:, :3].astype(np.float32)
    return world, (u, v)


def sample_rgb01(img_bgr: np.ndarray, uv):
    u, v = uv
    if len(u) == 0:
        return np.zeros((0, 3), np.float32)
    bgr = img_bgr[v, u, :3].astype(np.float32)
    rgb = bgr[:, ::-1] / 255.0
    return rgb


# ============================================================
# 3) 写 PLY（可选）
# ============================================================
def save_ply_xyzrgb(points: np.ndarray, colors_rgb01: np.ndarray | None, path: str):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors_rgb01 is not None and len(colors_rgb01) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors_rgb01, 0.0, 1.0).astype(np.float64))
    o3d.io.write_point_cloud(path, pcd)
    print("[OK] saved:", path, "points:", len(points))


def voxel_downsample_xyzrgb(points: np.ndarray, colors_rgb01: np.ndarray, voxel_size=0.005):
    if len(points) == 0:
        return points, colors_rgb01
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors_rgb01 is not None and len(colors_rgb01) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors_rgb01, 0.0, 1.0).astype(np.float64))
    pcd_down = pcd.voxel_down_sample(voxel_size=float(voxel_size))
    pts_down = np.asarray(pcd_down.points, dtype=np.float32)
    cols_down = np.asarray(pcd_down.colors, dtype=np.float32) if pcd_down.has_colors() else np.zeros((len(pts_down), 3), np.float32)
    return pts_down, cols_down


# ============================================================
# 4) 生成目标 JSON
# ============================================================
def colors01_to_u8(colors01: np.ndarray) -> np.ndarray:
    if colors01.size == 0:
        return colors01.astype(np.uint8)
    c = np.clip(colors01, 0.0, 1.0)
    return (c * 255.0 + 0.5).astype(np.uint8)


def build_target_json(
    object_pc: np.ndarray,
    object_pc_color01: np.ndarray,
    grasp_poses: np.ndarray | None,
    grasp_conf: np.ndarray | None,
    img_depth0: np.ndarray,
    img_color0_bgr: np.ndarray,
    full_pc0: np.ndarray,
    obj_mask0_u8: np.ndarray,
):
    grasp_poses_list = [] if grasp_poses is None else grasp_poses.tolist()
    grasp_conf_list = [] if grasp_conf is None else grasp_conf.tolist()
    pc_color_u8 = colors01_to_u8(object_pc_color01)

    return {
        "object_info": {
            "pc": object_pc.astype(np.float32).tolist(),
            "pc_color": pc_color_u8.astype(np.float32).tolist(),
        },
        "grasp_info": {
            "grasp_poses": grasp_poses_list,
            "grasp_conf": grasp_conf_list,
        },
        "scene_info": {
            "img_depth": img_depth0.astype(np.float32).tolist(),
            "img_color": img_color0_bgr.astype(np.uint8).tolist(),  # 维持你原来的写法（BGR）
            "full_pc": [full_pc0.astype(np.float32).tolist()],
            "obj_mask": obj_mask0_u8.astype(np.uint8).tolist(),
        }
    }


# ============================================================
# 5) 相机参数读取（json list）
# ============================================================
def load_cameras(camera_json_path: str):
    with open(camera_json_path, "r", encoding="utf-8") as f:
        cams = json.load(f)
    Ks = [np.asarray(c["intrinsics"], dtype=np.float32) for c in cams]
    w2cs = [np.asarray(c["extrinsics"], dtype=np.float32) for c in cams]
    return cams, Ks, w2cs


# ============================================================
# 6) 主流程：离线读取 depth/rgb/camjson/mask -> 输出 scene.json
# ============================================================
def main(
    rgb_dir: str,
    depth_dir: str,
    camera_json_path: str,
    mask_dir: str,
    export_dir: str,
    mask_mode: str = "rgb_nonblack",   # 你的mask类型
    nonblack_thresh: int = 10,
    invert_mode: str = "keep",
    voxel_size: float = 0.005,
    bin_thresh: int = 0,
    morph_close: int = 0,
    morph_open: int = 0,
):
    os.makedirs(export_dir, exist_ok=True)

    cams, Ks, w2cs = load_cameras(camera_json_path)

    # 以 depth_*.npy 文件为准决定要处理哪些 idx（避免 json 有很多条但目录只有 0000）
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "depth_*.npy")))
    if len(depth_files) == 0:
        raise FileNotFoundError(f"No depth_*.npy found in {depth_dir}")

    indices = []
    for p in depth_files:
        name = os.path.basename(p)  # depth_0000.npy
        idx = int(name.split("_")[1].split(".")[0])
        indices.append(idx)

    indices = sorted(indices)
    max_idx = max(indices)
    if max_idx >= len(cams):
        raise ValueError(f"camera_params.json has {len(cams)} entries but need at least idx={max_idx}")

    rgb_paths = {i: os.path.join(rgb_dir,  f"rgb_{i:04d}.png")  for i in indices}
    dep_paths = {i: os.path.join(depth_dir,f"depth_{i:04d}.npy")for i in indices}
    msk_paths = {i: os.path.join(mask_dir, f"mask_{i:04d}.png") for i in indices}

    print("[INFO] using indices:", indices)

    # ---------- 多视角 object 点云 ----------
    all_pts, all_cols = [], []
    for i in indices:
        rgb = cv2.imread(rgb_paths[i], cv2.IMREAD_COLOR)
        if rgb is None:
            print("[WARN] missing rgb:", rgb_paths[i])
            continue

        depth = np.load(dep_paths[i]).astype(np.float32)
        if depth.ndim == 1:
            Hr, Wr = rgb.shape[:2]
            if depth.size == Hr * Wr:
                depth = depth.reshape(Hr, Wr)
            else:
                raise ValueError(f"[view {i}] 1D depth size {depth.size} cannot reshape to rgb {Hr}x{Wr}")
        if depth.ndim != 2:
            raise ValueError(f"[view {i}] depth must be HxW, got {depth.shape}")

        H, W = depth.shape

        # rgb 统一到 depth 分辨率，避免 uv 采色错位/越界
        if rgb.shape[:2] != (H, W):
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)

        mask_img = cv2.imread(msk_paths[i], cv2.IMREAD_UNCHANGED)
        if mask_img is None:
            print("[WARN] missing mask:", msk_paths[i])
            continue

        # 你的 mask 是 RGB 黑底图，所以按非黑像素生成 bool mask
        mask = warp_mask_to_depth(
            mask_img,
            (H, W),
            mode=mask_mode,
            nonblack_thresh=nonblack_thresh,
            bin_thresh=bin_thresh,
            morph_close=morph_close,
            morph_open=morph_open,
        )
        mask = maybe_invert_mask(mask, mode=invert_mode)

        # debug：确认 mask 有效
        depth_ok = np.isfinite(depth) & (depth > 1e-6)
        print(
            f"[view {i}] depth shape={depth.shape} "
            f"min/max={float(np.nanmin(depth))}/{float(np.nanmax(depth))} "
            f"depth_ok={int(depth_ok.sum())} mask_on={int(mask.sum())} valid={int((mask & depth_ok).sum())}"
        )
        dbg_dir = os.path.join(export_dir, "debug")
        os.makedirs(dbg_dir, exist_ok=True)
        cv2.imwrite(os.path.join(dbg_dir, f"mask_aligned_{i:04d}.png"), (mask.astype(np.uint8) * 255))

        pts, uv = backproject_points_with_uv(depth, Ks[i], w2cs[i], mask)
        if len(pts) == 0:
            continue
        cols = sample_rgb01(rgb, uv)

        all_pts.append(pts)
        all_cols.append(cols)

    if not all_pts:
        raise RuntimeError("No points recovered. Check mask generation / mask file content.")

    obj_pts = np.concatenate(all_pts, axis=0).astype(np.float32)
    obj_cols = np.concatenate(all_cols, axis=0).astype(np.float32)
    print("[OBJ] pc:", obj_pts.shape, "pc_color:", obj_cols.shape)

    save_ply_xyzrgb(obj_pts, obj_cols, os.path.join(export_dir, "object_pc_rgb.ply"))
    obj_pts_down, obj_cols_down = voxel_downsample_xyzrgb(obj_pts, obj_cols, voxel_size=voxel_size)
    save_ply_xyzrgb(obj_pts_down, obj_cols_down, os.path.join(export_dir, "object_pc_rgb_down.ply"))

    # ---------- 第0视角 scene_info ----------
    i0 = indices[0]
    rgb0 = cv2.imread(rgb_paths[i0], cv2.IMREAD_COLOR)
    if rgb0 is None:
        raise FileNotFoundError(rgb_paths[i0])

    depth0 = np.load(dep_paths[i0]).astype(np.float32)
    if depth0.ndim == 1:
        Hr, Wr = rgb0.shape[:2]
        if depth0.size == Hr * Wr:
            depth0 = depth0.reshape(Hr, Wr)
        else:
            raise ValueError(f"[view {i0}] 1D depth size {depth0.size} cannot reshape to rgb {Hr}x{Wr}")
    if depth0.ndim != 2:
        raise ValueError(f"[view {i0}] depth must be HxW, got {depth0.shape}")

    H0, W0 = depth0.shape
    if rgb0.shape[:2] != (H0, W0):
        rgb0 = cv2.resize(rgb0, (W0, H0), interpolation=cv2.INTER_AREA)

    mask0_img = cv2.imread(msk_paths[i0], cv2.IMREAD_UNCHANGED)
    if mask0_img is None:
        raise FileNotFoundError(msk_paths[i0])

    obj_mask0 = warp_mask_to_depth(
        mask0_img,
        (H0, W0),
        mode=mask_mode,
        nonblack_thresh=nonblack_thresh,
        bin_thresh=bin_thresh,
        morph_close=morph_close,
        morph_open=morph_open,
    )
    obj_mask0 = maybe_invert_mask(obj_mask0, mode=invert_mode)
    obj_mask0_u8 = obj_mask0.astype(np.uint8)

    full_mask0 = np.ones((H0, W0), dtype=bool)
    full_pc0, _ = backproject_points_with_uv(depth0, Ks[i0], w2cs[i0], full_mask0)
    print("[SCENE] depth0:", depth0.shape, "rgb0:", rgb0.shape, "full_pc0:", full_pc0.shape)

    # ---------- grasp_info 先留空 ----------
    grasp_poses = None
    grasp_conf = None

    data = build_target_json(
        object_pc=obj_pts,
        object_pc_color01=obj_cols,
        grasp_poses=grasp_poses,
        grasp_conf=grasp_conf,
        img_depth0=depth0,
        img_color0_bgr=rgb0,
        full_pc0=full_pc0,
        obj_mask0_u8=obj_mask0_u8,
    )

    out_json = os.path.join(export_dir, "scene.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print("[OK] saved:", out_json)


if __name__ == "__main__":
    main(
        rgb_dir="assets/input/inputdata",
        depth_dir="assets/input/inputdata",
        camera_json_path="assets/input/camera_params.json",
        mask_dir="assets/input/inputdata",
        export_dir="./output_offline",
        mask_mode="rgb_nonblack",   # 你的mask：黑底RGB
        nonblack_thresh=10,         # 如果物体比较暗可调小到 3-5
        invert_mode="keep",         # 先 keep，别 auto
        voxel_size=0.005,
        morph_close=0,              # 例如 5：填小洞
        morph_open=0,               # 例如 3：去噪点
    )