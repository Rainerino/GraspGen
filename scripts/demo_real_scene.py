# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import glob
import json
import os
import time

import numpy as np
import omegaconf
import torch
import trimesh.transformations as tra
from tqdm import tqdm

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.utils.meshcat_utils import (
    create_visualizer,
    get_color_from_score,
    get_normals_from_mesh,
    make_frame,
    visualize_grasp,
    visualize_mesh,
    visualize_pointcloud,
    load_visualization_gripper_points,
)
from grasp_gen.utils.point_cloud_utils import (
    point_cloud_outlier_removal_with_color,
    filter_colliding_grasps,
)
from grasp_gen.robot import get_gripper_info

import meshcat.geometry as g
from grasp_gen.utils.meshcat_utils import rgb2hex


CTRL_PTS = [
    [0.06801729, 0.0, 0.195],
    [0.06801729, 0.0, 0.0975],
    [0.0, 0.0, 0.0975],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0975],
    [-0.06801729, 0.0, 0.0975],
    [-0.06801729, 0.0, 0.195],
]


def process_grasps_for_visualization(pc, grasps, grasp_conf, pc_colors=None):
    """Process grasps and point cloud for visualization by centering them."""
    scores = get_color_from_score(grasp_conf, use_255_scale=True)
    print(f"Scores with min {grasp_conf.min():.3f} and max {grasp_conf.max():.3f}")

    # Ensure grasps have correct homogeneous coordinate
    grasps[:, 3, 3] = 1

    # Center point cloud and grasps
    T_subtract_pc_mean = tra.translation_matrix(-pc.mean(axis=0))
    pc_centered = tra.transform_points(pc, T_subtract_pc_mean)
    grasps_centered = np.array([T_subtract_pc_mean @ np.array(g) for g in grasps.tolist()])

    return pc_centered, grasps_centered, scores, T_subtract_pc_mean


def viz_point(vis, name: str, p_xyz: np.ndarray, color=(0, 0, 255), radius=0.01):
    """Draw a colored sphere at p_xyz."""
    if vis is None:
        return
    vis[name].set_object(
        g.Sphere(radius),
        g.MeshBasicMaterial(color=rgb2hex(tuple(color))),
    )
    T = np.eye(4, dtype=float)
    T[:3, 3] = p_xyz.astype(float)
    vis[name].set_transform(T)


def viz_segment(vis, name, p0, p1, color=(0, 255, 255), linewidth=4):
    """Draw a line segment from p0 to p1 in WORLD coordinates."""
    p0 = np.asarray(p0, dtype=np.float32).reshape(3)
    p1 = np.asarray(p1, dtype=np.float32).reshape(3)

    P = np.stack([p0, p1], axis=1)  # (3,2)
    P = np.vstack([P, np.ones((1, 2), dtype=np.float32)])  # (4,2) homogeneous for meshcat

    vis[name].set_object(
        g.Line(
            g.PointsGeometry(P),
            g.MeshBasicMaterial(
                color=int(color[0]) * 256 * 256 + int(color[1]) * 256 + int(color[2]),
                linewidth=linewidth,
            ),
        )
    )


# =============================
# Save grasp transforms helpers
# =============================
def make_T_from_R_p(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """R(3,3) + p(3,) -> T(4,4)"""
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(p, dtype=float).reshape(3)
    return T


def save_grasp_Ts_json(out_path: str, Ts: list[np.ndarray], scores: np.ndarray | None = None):
    """
    Save list of 4x4 transforms to json.
    Format:
      [{"id": i, "T_4x4": [[...],[...],[...],[...]], "score": ...}, ...]
    """
    payload = []
    for i, T in enumerate(Ts):
        item = {
            "id": i,
            "T_4x4": np.asarray(T, dtype=float).tolist(),
        }
        if scores is not None and i < len(scores):
            item["score"] = float(scores[i])
        payload.append(item)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved {len(payload)} grasp poses -> {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize grasps on a scene point cloud after GraspGen inference, for entire scene"
    )
    parser.add_argument(
        "--sample_data_dir",
        type=str,
        default="",
        help="Directory containing JSON files with point cloud data",
    )
    parser.add_argument(
        "--gripper_config",
        type=str,
        default="",
        help="Path to gripper configuration YAML file",
    )
    parser.add_argument(
        "--grasp_threshold",
        type=float,
        default=0.60,
        help="Threshold for valid grasps. If -1.0, then the top 100 grasps will be ranked and returned",
    )
    parser.add_argument(
        "--num_grasps",
        type=int,
        default=1,
        help="Number of grasps to generate",
    )
    parser.add_argument(
        "--return_topk",
        action="store_true",
        help="Whether to return only the top k grasps",
    )
    parser.add_argument(
        "--topk_num_grasps",
        type=int,
        default=-1,
        help="Number of top grasps to return when return_topk is True",
    )
    parser.add_argument(
        "--filter_collisions",
        action="store_true",
        help="Whether to filter grasps based on collision detection with scene point cloud",
    )
    parser.add_argument(
        "--collision_threshold",
        type=float,
        default=0.01,
        help="Distance threshold for collision detection (in meters)",
    )
    parser.add_argument(
        "--max_scene_points",
        type=int,
        default=8192,
        help="Maximum number of scene points to use for collision checking (for speed optimization)",
    )

    # NEW: output path for saving grasp transforms
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="Output directory or json file path to save grasp 4x4 transforms. "
             "If directory, a per-scene json will be created there. If empty, defaults to sample_data_dir.",
    )
    parser.add_argument(
        "--save_translation",
        type=str,
        default="p_world",
        choices=["p_world", "t"],
        help="Which translation to use in saved 4x4: "
             "'p_world' (your grasp point) or 't' (grasp origin).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.sample_data_dir == "":
        raise ValueError("sample_data_dir is required")
    if args.gripper_config == "":
        raise ValueError("gripper_config is required")

    if not os.path.exists(args.sample_data_dir):
        raise FileNotFoundError(f"sample_data_dir {args.sample_data_dir} does not exist")

    # Handle return_topk logic
    if args.return_topk and args.topk_num_grasps == -1:
        args.topk_num_grasps = 100

    json_files = glob.glob(os.path.join(args.sample_data_dir, "*.json"))
    if len(json_files) == 0:
        raise FileNotFoundError(f"No .json files found in {args.sample_data_dir}")

    # Load gripper config and get gripper name
    grasp_cfg = load_grasp_cfg(args.gripper_config)
    gripper_name = grasp_cfg.data.gripper_name

    # Get gripper collision mesh for collision filtering
    gripper_info = get_gripper_info(gripper_name)
    gripper_collision_mesh = gripper_info.collision_mesh
    print(f"Using gripper: {gripper_name}")
    print(f"Gripper collision mesh has {len(gripper_collision_mesh.vertices)} vertices")

    # Initialize GraspGenSampler once
    grasp_sampler = GraspGenSampler(grasp_cfg)

    vis = create_visualizer()

    for json_file in json_files:
        print(json_file)
        vis.delete()

        data = json.load(open(json_file, "rb"))

        obj_pc = np.array(data["object_info"]["pc"])
        obj_pc_color = np.array(data["object_info"]["pc_color"])

        full_pc_key = "pc_color" if "pc_color" in data["scene_info"] else "full_pc"
        xyz_scene = np.array(data["scene_info"][full_pc_key])[0]
        xyz_scene_color = np.array(data["scene_info"]["img_color"]).reshape(1, -1, 3)[0, :, :]

        # Remove object points from scene point cloud (mask == 1 is object)
        xyz_seg = np.array(data["scene_info"]["obj_mask"]).reshape(-1)
        xyz_scene = xyz_scene[xyz_seg != 1]
        xyz_scene_color = xyz_scene_color[xyz_seg != 1]

        VIZ_BOUNDS = [[-1.5, -1.25, -0.15], [1.5, 1.25, 2.0]]
        mask_within_bounds = np.all((xyz_scene > VIZ_BOUNDS[0]), 1)
        mask_within_bounds = np.logical_and(mask_within_bounds, np.all((xyz_scene < VIZ_BOUNDS[1]), 1))

        xyz_scene = xyz_scene[mask_within_bounds]
        xyz_scene_color = xyz_scene_color[mask_within_bounds]

        visualize_pointcloud(vis, "pc_scene", xyz_scene, xyz_scene_color, size=0.0025)

        obj_pc, pc_removed, obj_pc_color, obj_pc_color_removed = point_cloud_outlier_removal_with_color(
            torch.from_numpy(obj_pc), torch.from_numpy(obj_pc_color)
        )
        obj_pc = obj_pc.cpu().numpy()
        obj_pc_color = obj_pc_color.cpu().numpy()

        visualize_pointcloud(vis, "pc_obj", obj_pc, obj_pc_color, size=0.005)

        grasps, grasp_conf = GraspGenSampler.run_inference(
            obj_pc,
            grasp_sampler,
            grasp_threshold=args.grasp_threshold,
            num_grasps=args.num_grasps,
            topk_num_grasps=args.topk_num_grasps,
        )

        if len(grasps) > 0:
            grasp_conf = grasp_conf.cpu().numpy()
            grasps = grasps.cpu().numpy()
            grasps[:, 3, 3] = 1

            # Process grasps for visualization (centering)
            obj_pc_centered, grasps_centered, scores, T_center = process_grasps_for_visualization(
                obj_pc, grasps, grasp_conf, obj_pc_color
            )

            # Center scene point cloud using same transformation
            xyz_scene_centered = tra.transform_points(xyz_scene, T_center)

            # Apply collision filtering if requested
            collision_free_grasps = grasps_centered
            collision_free_scores = scores

            if args.filter_collisions:
                print("Applying collision filtering...")
                collision_start = time.time()

                if len(xyz_scene_centered) > args.max_scene_points:
                    indices = np.random.choice(len(xyz_scene_centered), args.max_scene_points, replace=False)
                    xyz_scene_downsampled = xyz_scene_centered[indices]
                    print(f"Downsampled scene pc from {len(xyz_scene_centered)} to {len(xyz_scene_downsampled)}")
                else:
                    xyz_scene_downsampled = xyz_scene_centered
                    print(f"Scene pc has {len(xyz_scene_centered)} points (no downsampling)")

                collision_free_mask = filter_colliding_grasps(
                    scene_pc=xyz_scene_downsampled,
                    grasp_poses=grasps_centered,
                    gripper_collision_mesh=gripper_collision_mesh,
                    collision_threshold=args.collision_threshold,
                )

                collision_free_grasps = grasps_centered[collision_free_mask]
                collision_free_scores = scores[collision_free_mask]

                collision_time = time.time() - collision_start
                print(f"Collision detection took: {collision_time:.2f}s")
                print(f"Found {len(collision_free_grasps)}/{len(grasps_centered)} collision-free grasps")

            grasps_to_visualize = collision_free_grasps if args.filter_collisions else grasps_centered
            scores_to_use = collision_free_scores

            # ---- collect & save 4x4 transforms ----
            grasp_T_list = []

            for j, grasp in enumerate(grasps_to_visualize):
                color = scores_to_use[j] if not args.filter_collisions else [0, 185, 0]

                visualize_grasp(
                    vis,
                    f"grasps/{j:03d}/grasp",
                    tra.inverse_matrix(T_center) @ grasp,
                    color=color,
                    gripper_name=gripper_name,
                    linewidth=1.5,
                )

                T = tra.inverse_matrix(T_center) @ grasp
                R = T[:3, :3]
                t = T[:3, 3]

                p_local = np.array([0.0, 0.0, 0.195], dtype=float)
                p_world = R @ p_local + t

                print("grasp point world:", p_world)
                print("R:\n", R)

                # Save transform: choose translation
                if args.save_translation == "p_world":
                    T_4x4 = make_T_from_R_p(R, p_world)
                else:
                    T_4x4 = make_T_from_R_p(R, t)

                grasp_T_list.append(T_4x4)

                # Visualization extras
                viz_point(vis, f"grasps/{j:03d}/grasp_point", p_world, color=(255, 255, 0), radius=0.01)
                viz_segment(
                    vis,
                    f"grasps/{j:03d}/grasp_dir",
                    t,
                    p_world,
                    color=(0, 255, 255),
                    linewidth=6,
                )

            # Decide output path
            base_name = os.path.basename(json_file).replace(".json", "")
            if args.output_path == "":
                out_dir_or_file = args.sample_data_dir
            else:
                out_dir_or_file = args.output_path

            # If output_path ends with .json => write exactly there (single scene expected)
            # Otherwise treat as directory and write per-scene json inside it.
            if out_dir_or_file.lower().endswith(".json"):
                out_pose_json = out_dir_or_file
            else:
                os.makedirs(out_dir_or_file, exist_ok=True)
                out_pose_json = os.path.join(out_dir_or_file, f"grasps_T_4x4_{base_name}.json")

            save_grasp_Ts_json(out_pose_json, grasp_T_list, scores=np.asarray(grasp_conf))

            input("Press Enter to continue to next scene...")
        else:
            print("No grasps found! Skipping to next scene...")