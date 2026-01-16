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
      - binary：2D mask [H, W]，值为 1 的像素表示object区域（保留），其他像素被mask掉（排除）
      - auto：自动判断（3通道->rgb_nonblack, 2D->binary, 其他->gray）

    返回: (H_out, W_out) bool，True表示保留的点（object区域），False表示被mask掉的点
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
        elif mask_rs.ndim == 2:
            mode = "binary"
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
    
    elif mode == "binary":
        # 处理 [H, W] 形状的 mask，值为 1 的像素表示object区域（保留），其他像素被mask掉（排除）
        if mask_rs.ndim != 2:
            raise ValueError(f"binary mode expects 2D mask [H, W], got {mask_rs.shape}")
        
        # 转换为 uint8 以便处理
        if mask_rs.dtype == np.uint16:
            mask_u8 = (mask_rs / 257).astype(np.uint8)
        else:
            mask_u8 = mask_rs.astype(np.uint8)
        
        # 检查mask的值范围，用于调试
        unique_vals = np.unique(mask_u8)
        print(f"[DEBUG] binary mask unique values: {unique_vals}, shape: {mask_u8.shape}, min={mask_u8.min()}, max={mask_u8.max()}")
        
        # 值为 1 的像素表示object区域（保留），其他像素被mask掉（排除）
        # 如果mask中没有值为1，检查是否有其他明显的object值（比如255）
        if 1 in unique_vals:
            # 值为1的地方是object区域，应该保留
            mb = (mask_u8 == 1)
            print(f"[DEBUG] binary mask: treating value 1 as object region (to be kept)")
        elif 255 in unique_vals and len(unique_vals) <= 3:
            # 如果mask是二值图（0和255），值为255的地方是object区域
            mb = (mask_u8 == 255)
            print(f"[DEBUG] binary mask: treating 255 as object region (to be kept)")
        else:
            # 默认：值大于0的像素是object区域，值为0的像素被mask掉
            mb = (mask_u8 > 0)
            print(f"[DEBUG] binary mask: treating non-zero values as object region (to be kept)")
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
def fill_invalid_depth(depth: np.ndarray, default_value: float = 1e-5) -> np.ndarray:
    """
    将无效的 depth 值（NaN、inf 或 <= 1e-6）替换为默认值，确保所有像素都能生成点云。
    
    Args:
        depth: 2D depth 图像
        default_value: 用于替换无效值的默认值，默认 1e-5（需要大于 1e-6 以通过有效性检查）
    
    Returns:
        处理后的 depth 图像
    """
    depth_filled = depth.copy()
    invalid_mask = ~(np.isfinite(depth) & (depth > 1e-6))
    if invalid_mask.any():
        depth_filled[invalid_mask] = default_value
        print(f"[DEPTH FILL] Filled {invalid_mask.sum()} invalid depth pixels with {default_value}")
    return depth_filled


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
    print(f"cam_h: {cam_h.shape}")
    print(f"c2w: {c2w.shape}")
    world = (c2w @ cam_h.T).T[:, :3].astype(np.float32)
    return world, (u, v)


def sample_rgb01(img_bgr: np.ndarray, uv):
    u, v = uv
    if len(u) == 0:
        return np.zeros((0, 3), np.float32)
    
    # 验证尺寸一致性
    H_img, W_img = img_bgr.shape[:2]
    u_max, v_max = u.max(), v.max()
    if u_max >= W_img or v_max >= H_img:
        raise ValueError(f"UV coordinates out of bounds: u_max={u_max} >= W={W_img}, v_max={v_max} >= H={H_img}")
    
    # 采样 RGB 颜色（v 是行索引，u 是列索引）
    bgr = img_bgr[v, u, :3].astype(np.float32)
    rgb = bgr[:, ::-1] / 255.0  # BGR -> RGB
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
    img_depth0: np.ndarray | None,
    img_color0_bgr: np.ndarray | None,
    full_pc0: np.ndarray,
    obj_mask0_u8: np.ndarray | None,
):
    grasp_poses_list = [] if grasp_poses is None else grasp_poses.tolist()
    grasp_conf_list = [] if grasp_conf is None else grasp_conf.tolist()
    pc_color_u8 = colors01_to_u8(object_pc_color01)

    scene_info = {
        "full_pc": [full_pc0.astype(np.float32).tolist()],
    }
    
    # 保存图像数据（这些数据很大，但需要保存）
    if img_depth0 is not None:
        scene_info["img_depth"] = img_depth0.astype(np.float32).tolist()
    if img_color0_bgr is not None:
        scene_info["img_color"] = img_color0_bgr.astype(np.uint8).tolist()  # 维持你原来的写法（BGR）
    if obj_mask0_u8 is not None:
        scene_info["obj_mask"] = obj_mask0_u8.astype(np.uint8).tolist()

    return {
        "object_info": {
            "pc": object_pc.astype(np.float32).tolist(),
            "pc_color": pc_color_u8.astype(np.float32).tolist(),
        },
        "grasp_info": {
            "grasp_poses": grasp_poses_list,
            "grasp_conf": grasp_conf_list,
        },
        "scene_info": scene_info,
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
    mask_mode: str = "auto",   # 你的mask类型
    nonblack_thresh: int = 10,
    invert_mode: str = "keep",
    voxel_size: float = 0.005,
    bin_thresh: int = 0,
    morph_close: int = 0,
    morph_open: int = 0,
    downsample_ratio: float = 1.0,  # 图像降采样比率（先对image和mask进行等比例放缩，然后再映射到点云），1.0表示不下采样，0.5表示保留50%的像素
    scene_downsample_ratio: float | None = None,  # scene_info中full_pc的降采样比率，None表示使用downsample_ratio
    # save_images_to_json: bool = True,  # 是否将img_depth、img_color、obj_mask保存到JSON（这些数据很大）
):
    os.makedirs(export_dir, exist_ok=True)

    cams, Ks, w2cs = load_cameras(camera_json_path)
    Ks_orig = [K.copy() for K in Ks]  # 保存原始内参，供 scene_info 使用

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
    processed_indices = []  # 记录成功处理的索引
    for i in indices:
        rgb = cv2.imread(rgb_paths[i], cv2.IMREAD_COLOR)
        if rgb is None:
            print("[WARN] missing rgb:", rgb_paths[i])
            continue

        depth = np.load(dep_paths[i]).astype(np.float32)
        # 处理 3D depth (H, W, 1) 或 (H, W, C)
        if depth.ndim == 3:
            if depth.shape[2] == 1:
                depth = depth[:, :, 0]  # 压缩单通道维度
            else:
                depth = depth[:, :, 0]  # 如果有多个通道，取第一个
        # 处理 1D depth
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
        # 使用 INTER_LINEAR 保持质量同时减少对齐误差（如果要求严格对齐，可使用 INTER_NEAREST）
        if rgb.shape[:2] != (H, W):
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)

        mask_img = cv2.imread(msk_paths[i], cv2.IMREAD_UNCHANGED)
        if mask_img is None:
            print("[WARN] missing mask:", msk_paths[i])
            continue

        # 图像空间降采样：先对 image 和 mask 进行等比例放缩，然后再映射到点云
        K_orig = Ks_orig[i].copy()  # 使用原始内参
        if downsample_ratio < 1.0 and downsample_ratio > 0.0:
            # 计算新的图像尺寸
            H_new = int(H * np.sqrt(downsample_ratio))
            W_new = int(W * np.sqrt(downsample_ratio))
            # 确保尺寸为正且不超出原始尺寸
            H_new = max(1, min(H_new, H))
            W_new = max(1, min(W_new, W))
            
            # 等比例缩放 depth、rgb、mask
            # 注意：为了确保 RGB 和 depth 像素位置完全对齐，rgb 也使用 INTER_NEAREST
            if (H_new, W_new) != (H, W):
                depth = cv2.resize(depth, (W_new, H_new), interpolation=cv2.INTER_NEAREST)
                rgb = cv2.resize(rgb, (W_new, H_new), interpolation=cv2.INTER_NEAREST)  # 使用 NEAREST 确保与 depth 对齐
                mask_img = cv2.resize(mask_img, (W_new, H_new), interpolation=cv2.INTER_NEAREST)
            
            # 调整内参 K（fx, fy, cx, cy 都需要乘以缩放比例）
            scale_h = H_new / H
            scale_w = W_new / W
            K_new = K_orig.copy()
            K_new[0, 0] *= scale_w  # fx
            K_new[1, 1] *= scale_h  # fy
            K_new[0, 2] *= scale_w  # cx
            K_new[1, 2] *= scale_h  # cy
            Ks[i] = K_new
            H, W = H_new, W_new
            print(f"[view {i}] image downsampled to {H}x{W} (ratio={downsample_ratio:.2f}, scale={scale_h:.3f}x{scale_w:.3f})")

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
        processed_indices.append(i)  # 记录成功处理的索引

    if not all_pts:
        raise RuntimeError("No points recovered. Check mask generation / mask file content.")

    obj_pts = np.concatenate(all_pts, axis=0).astype(np.float32)
    obj_cols = np.concatenate(all_cols, axis=0).astype(np.float32)
    print("[OBJ] pc (after image-space downsample):", obj_pts.shape, "pc_color:", obj_cols.shape)

    save_ply_xyzrgb(obj_pts, obj_cols, os.path.join(export_dir, "object_pc_rgb.ply"))
    obj_pts_down, obj_cols_down = voxel_downsample_xyzrgb(obj_pts, obj_cols, voxel_size=voxel_size)
    save_ply_xyzrgb(obj_pts_down, obj_cols_down, os.path.join(export_dir, "object_pc_rgb_down.ply"))

    # ---------- 第0视角 scene_info ----------
    # 使用第一个成功处理的索引，如果没有则使用原始索引列表的第一个
    if not processed_indices:
        raise RuntimeError("No valid views processed. Cannot generate scene_info.")
    i0 = processed_indices[0]
    if i0 not in rgb_paths:
        raise KeyError(f"Index {i0} not found in rgb_paths. Available indices: {list(rgb_paths.keys())}")
    rgb0 = cv2.imread(rgb_paths[i0], cv2.IMREAD_COLOR)
    if rgb0 is None:
        raise FileNotFoundError(rgb_paths[i0])

    depth0 = np.load(dep_paths[i0]).astype(np.float32)
    # 处理 3D depth (H, W, 1) 或 (H, W, C)
    if depth0.ndim == 3:
        if depth0.shape[2] == 1:
            depth0 = depth0[:, :, 0]  # 压缩单通道维度
        else:
            depth0 = depth0[:, :, 0]  # 如果有多个通道，取第一个
    # 处理 1D depth
    if depth0.ndim == 1:
        Hr, Wr = rgb0.shape[:2]
        if depth0.size == Hr * Wr:
            depth0 = depth0.reshape(Hr, Wr)
        else:
            raise ValueError(f"[view {i0}] 1D depth size {depth0.size} cannot reshape to rgb {Hr}x{Wr}")
    if depth0.ndim != 2:
        raise ValueError(f"[view {i0}] depth must be HxW, got {depth0.shape}")

    H0_orig, W0_orig = depth0.shape  # 保存原始尺寸
    print(f"[SCENE] Original depth0 shape: {H0_orig}x{W0_orig} = {H0_orig * W0_orig} pixels")
    H0, W0 = H0_orig, W0_orig
    # rgb 统一到 depth 分辨率，使用 INTER_LINEAR 保持质量同时减少对齐误差
    if rgb0.shape[:2] != (H0, W0):
        rgb0 = cv2.resize(rgb0, (W0, H0), interpolation=cv2.INTER_LINEAR)

    mask0_img = cv2.imread(msk_paths[i0], cv2.IMREAD_UNCHANGED)
    if mask0_img is None:
        raise FileNotFoundError(msk_paths[i0])

    # scene_info 的图像空间降采样（使用原始内参）
    K0_orig = Ks_orig[i0].copy()
    scene_ratio = scene_downsample_ratio if scene_downsample_ratio is not None else downsample_ratio
    if scene_ratio < 1.0 and scene_ratio > 0.0:
        # 计算新的图像尺寸
        H0_new = int(H0 * np.sqrt(scene_ratio))
        W0_new = int(W0 * np.sqrt(scene_ratio))
        # 确保尺寸为正且不超出原始尺寸
        H0_new = max(1, min(H0_new, H0))
        W0_new = max(1, min(W0_new, W0))
        
        # 等比例缩放 depth0、rgb0、mask0_img
        # 注意：为了确保 RGB 和 depth 像素位置完全对齐，rgb0 也使用 INTER_NEAREST
        if (H0_new, W0_new) != (H0, W0):
            # 检查降采样前的有效 depth 值数量
            depth_valid_before = np.isfinite(depth0) & (depth0 > 1e-6)
            print(f"[SCENE] Before downsample: depth shape {H0}x{W0}, valid pixels: {depth_valid_before.sum()}/{depth0.size}")
            depth0 = cv2.resize(depth0, (W0_new, H0_new), interpolation=cv2.INTER_NEAREST)
            rgb0 = cv2.resize(rgb0, (W0_new, H0_new), interpolation=cv2.INTER_NEAREST)  # 使用 NEAREST 确保与 depth 对齐
            mask0_img = cv2.resize(mask0_img, (W0_new, H0_new), interpolation=cv2.INTER_NEAREST)
        # 检查降采样后的有效 depth 值数量
        depth_valid_after = np.isfinite(depth0) & (depth0 > 1e-6)
        print(f"[SCENE] After downsample: depth shape {H0_new}x{W0_new}, valid pixels: {depth_valid_after.sum()}/{depth0.size}")
        
        # 调整内参 K
        scale_h0 = H0_new / H0_orig  # 使用原始尺寸计算缩放比例
        scale_w0 = W0_new / W0_orig
        K0_new = K0_orig.copy()
        K0_new[0, 0] *= scale_w0  # fx
        K0_new[1, 1] *= scale_h0  # fy
        K0_new[0, 2] *= scale_w0  # cx
        K0_new[1, 2] *= scale_h0  # cy
        H0, W0 = H0_new, W0_new
        print(f"[SCENE] image downsampled to {H0}x{W0} (ratio={scene_ratio:.2f}, scale={scale_h0:.3f}x{scale_w0:.3f})")
    else:
        K0_new = K0_orig

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
    # 检查 depth0 的有效值
    depth_valid = np.isfinite(depth0) & (depth0 > 1e-6)
    n_invalid = depth0.size - depth_valid.sum()
    print(f"[SCENE] depth0 shape: {depth0.shape}, total pixels: {depth0.size}, valid: {depth_valid.sum()}, invalid: {n_invalid}")
    
    # 将无效的 depth 值替换为默认值（1e-5，略大于 1e-6 以通过有效性检查），确保所有像素都能生成点云
    if n_invalid > 0:
        depth0_filled = fill_invalid_depth(depth0, default_value=1e-5)
    else:
        depth0_filled = depth0
    
    full_pc0, _ = backproject_points_with_uv(depth0_filled, K0_new, w2cs[i0], full_mask0)
    print("[SCENE] depth0:", depth0.shape, "rgb0:", rgb0.shape, "full_pc0 (after image-space downsample):", full_pc0.shape)
    expected_points = H0 * W0
    actual_points = len(full_pc0)
    if expected_points != actual_points:
        print(f"[SCENE WARN] Expected points: {expected_points}, Actual points: {actual_points}, Difference: {expected_points - actual_points}")
    else:
        print(f"[SCENE OK] Point count matches: {actual_points} = {H0} * {W0}")

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
        export_dir="/home/xuran-yao/code/june_serve/GraspGen/GraspGenModels/sample_data/real_scene_pc/real_scene",
        mask_mode="auto",   # 你的mask：黑底RGB
        nonblack_thresh=10,         # 如果物体比较暗可调小到 3-5
        invert_mode="keep",         # 先 keep，别 auto
        voxel_size=0.005,
        morph_close=0,              # 例如 5：填小洞
        morph_open=0,               # 例如 3：去噪点
        downsample_ratio=0.1,       # 点云下采样比率，1.0表示不下采样，0.5表示保留50%的点
        scene_downsample_ratio=1, # scene_info中full_pc的下采样比率
        # save_images_to_json=False,   # 是否将img_depth、img_color、obj_mask保存到JSON（这些数据很大，建议False）
    )