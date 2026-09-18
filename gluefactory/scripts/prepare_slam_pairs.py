#!/usr/bin/env python3
"""
Generate training/validation image pair lists for the SLAM dataset.

Uses camera poses from the SLAM trajectory to find geometrically overlapping
image pairs: two frames are paired if their translational distance falls within
[--min_dist, --max_dist] and their co-visibility (estimated from depth overlap)
exceeds a threshold. Writes pairs_train.txt and pairs_val.txt in the dataset
directory, one pair per line formatted as "rgb/frame_a.png rgb/frame_b.png".

Usage:
    python -m gluefactory.scripts.prepare_slam_pairs \\
        --data_dir    data/output/slam \\
        --poses_file  data/output/slam/poses.txt \\
        --modality    rgb \\
        --min_dist    0.05 \\
        --max_dist    1.0  \\
        --val_ratio   0.15
"""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from gluefactory.slam.geometry import (
    compute_covisibility,
    get_pose_index,
    load_camera_intrinsics,
    match_image_timestamp,
    parse_poses,
)
from gluefactory.slam.extractor import SLAMFeatureExtractor
from gluefactory.slam.io import write_h5_features
from gluefactory.visualization.datasetviz import visualize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate Pose-Based Pairs for SLAM")
    parser.add_argument("--data_dir", type=str, default="data/output/sample_slam")
    parser.add_argument("--max_dist", type=float, default=2.0)
    parser.add_argument("--min_dist", type=float, default=0.1)
    parser.add_argument("--max_angle", type=float, default=30.0)
    parser.add_argument("--min_overlap", type=float, default=0.1)
    parser.add_argument("--max_pairs", type=int, default=10)
    parser.add_argument("--split_ratio", type=float, default=0.83)
    parser.add_argument("--extract_features", action="store_true", default=False)
    parser.add_argument("--sp_weights", type=str, default=None)
    parser.add_argument("--nms_radius", type=int, default=3)
    parser.add_argument("--max_keypoints", type=int, default=512)
    parser.add_argument("--num_threads", type=int, default=14)
    parser.add_argument("--num_vis", type=int, default=50,
                        help="Number of pairs to visualize automatically after "
                             "creation (0 = skip).")
    args = parser.parse_args()

    dataset_dir = Path(args.data_dir)
    poses_path = dataset_dir / "poses_odom_RGBD_slam.txt"
    if not poses_path.exists():
        logger.error(f"Poses file not found: {poses_path}")
        return

    rgb_dir = dataset_dir / "images/rgb"
    depth_dir = dataset_dir / "images/depth"
    calib_dir = dataset_dir / "images/calib"

    image_names = sorted(
        [p.name for p in rgb_dir.glob("*.png")] + [p.name for p in rgb_dir.glob("*.jpg")]
    )
    if not image_names:
        logger.error("No images found.")
        return

    poses = parse_poses(poses_path)

    first_ts = Path(image_names[0]).stem
    calib_path = calib_dir / f"{first_ts}.yaml"
    if not calib_path.exists():
        calib_path = list(calib_dir.glob("*.yaml"))[0]
    K = load_camera_intrinsics(calib_path)

    logger.info("Computing pairwise distances and covisibility...")
    pairs = []

    pose_idx = get_pose_index(poses)
    pose_ts_sorted, pose_keys_sorted = pose_idx

    name_to_ts = {}
    valid_names = []
    for n in image_names:
        matched_key = match_image_timestamp(n, *pose_idx)
        if matched_key is None:
            continue
        name_to_ts[n] = matched_key
        valid_names.append(n)

    logger.info(f"Matched {len(valid_names)}/{len(image_names)} images to poses.")
    for n in valid_names[:3]:
        logger.debug(f"  {n} -> pose ts {name_to_ts[n]} (delta={abs(float(Path(n).stem.replace('_','.')) - float(name_to_ts[n])):.6f}s)")

    for i, name_i in enumerate(tqdm(valid_names)):
        ts_i = name_to_ts[name_i]
        R_i, t_i = poses[ts_i]

        candidates = []
        for j, name_j in enumerate(valid_names):
            if i == j:
                continue
            ts_j = name_to_ts[name_j]
            R_j, t_j = poses[ts_j]
            dist = np.linalg.norm(t_i - t_j)
            if dist < args.min_dist or dist > args.max_dist:
                continue
            R_rel = R_i.T @ R_j
            trace = np.trace(R_rel)
            angle = np.degrees(np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0)))
            if angle > args.max_angle:
                continue
            candidates.append((j, name_j, ts_j, R_j, t_j, dist))

        candidates.sort(key=lambda x: x[-1])
        candidates = candidates[: args.max_pairs * 2]

        valid_candidates = []
        depth_path_i = depth_dir / name_i
        for j, name_j, ts_j, R_j, t_j, dist in candidates:
            overlap = compute_covisibility(depth_path_i, R_i, t_i, R_j, t_j, K)
            if overlap >= args.min_overlap:
                valid_candidates.append((name_i, name_j))
                if len(valid_candidates) >= args.max_pairs:
                    break
        pairs.extend(valid_candidates)

    logger.info(f"Found {len(pairs)} pairs.")

    np.random.seed(42)
    np.random.shuffle(pairs)
    num_train = int(len(pairs) * args.split_ratio)
    train_pairs = pairs[:num_train]
    val_pairs = pairs[num_train:]

    with open(dataset_dir / "pairs_train.txt", "w") as f:
        for pi, pj in train_pairs:
            f.write(f"rgb/{pi} rgb/{pj}\n")
    with open(dataset_dir / "pairs_val.txt", "w") as f:
        for pi, pj in val_pairs:
            f.write(f"rgb/{pi} rgb/{pj}\n")

    if args.extract_features:
        logger.info("Extracting features...")
        exports_dir = dataset_dir / "exports"
        exports_dir.mkdir(exist_ok=True)
        h5_path = exports_dir / "sp_features_slam.h5"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        extractor = SLAMFeatureExtractor(
            args.sp_weights,
            {
                "nms_radius": args.nms_radius,
                "max_num_keypoints": args.max_keypoints,
                "detection_threshold": 0.0,
            },
            device,
        )
        unique_images = list(set([p[0] for p in pairs] + [p[1] for p in pairs]))

        def worker(name):
            out = extractor.extract(dataset_dir / "images/rgb" / name)
            return name, out["keypoints"], out["keypoint_scores"], out["descriptors"]

        with h5py.File(h5_path, "w") as f:
            with tqdm(total=len(unique_images)) as pbar:
                if args.num_threads > 1:
                    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
                        for name, kpts, scores, desc in executor.map(worker, unique_images):
                            if kpts is not None:
                                write_h5_features(f, name, kpts, scores, desc)
                            pbar.update(1)
                else:
                    for n in unique_images:
                        name, kpts, scores, desc = worker(n)
                        if kpts is not None:
                            write_h5_features(f, name, kpts, scores, desc)
                        pbar.update(1)

    logger.info("Done generating pairs and features.")

    if args.num_vis and args.num_vis > 0:
        logger.info(f"Auto-visualizing {args.num_vis} pairs after creation...")
        visualize(
            "pairs",
            data_dir=str(dataset_dir),
            h5_path=str(dataset_dir / "exports/sp_features_slam.h5")
            if (dataset_dir / "exports/sp_features_slam.h5").exists()
            else None,
            pairs_file="pairs_train.txt",
            num_vis=args.num_vis,
        )
        logger.info("Pair visualization complete.")


if __name__ == "__main__":
    main()
