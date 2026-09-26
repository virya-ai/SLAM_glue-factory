#!/usr/bin/env python3
"""
Build an end-to-end RGB-D SLAM dataset from a
RTAB-Map ``.db`` database.

Selection is config-driven and geometrically balanced:

1. **Frames** — choose keyframes out of the map.  The default ``bins`` mode
   bins every *consecutive-frame transition* into (translation, rotation)
   cells and samples each cell up to ``frame.samples_per_bin`` evenly;
   ``minmax`` gate-samples frames by ``frame.min_trans``/``frame.min_rot``;
   ``all`` keeps every node; ``stride`` keeps every Nth node.
2. **Pairs** — candidates come from the RTAB-Map graph links
   (``pairs.candidates: links``), geometric proximity (``spatial``) or all
   pairs (``all``); they are filtered with ``min/max_dist`` and
   ``min/max_angle`` (optionally ``min_overlap`` via depth covisibility),
   then **balanced** per (translation, rotation) bin with per-bin sample
   caps, ``max_pairs_per_image`` and ``max_pairs`` limits.
3. **Extraction** — the selected subset (rgb + depth + calib) is decoded
   from the DB blobs and written to ``data_dir``.

Poses stored in ``poses_odom_RGBD_slam.txt`` are **camera-optical-frame**
poses in the world: ``T_map2cam = T_base_link @ T_base2camera``, composed
from the RTAB-Map Node (base_link) pose and the calibration blob's
``local_transform`` (base -> camera optical frame) — the same composition
RTAB-Map's own Export tool applies
(``cameraViewpoint = nodePose * localTransform``).

Output layout (consumed by ``SlamPosedDataset`` / superpoint_slam):

    data_dir/
        poses_odom_RGBD_slam.txt
        image_list_train.txt / image_list_val.txt
        pairs_train.txt / pairs_val.txt
        images/rgb/<stem>.png          # decoded JPEG -> PNG (BGR)
        images/depth/<stem>.png        # uint16 mm
        images/calib/<stem>.yaml

``<stem>`` is the camera stamp as 9 decimals with ``.`` -> ``_`` so pose
keys and image stems match exactly.  Train/val are split on *frames* first,
so the two partitions never share an image.

Paths: ``--db`` and ``--data_dir`` (their config-equivalents and the built-in
defaults) are resolved relative to the workspace root
``PROJECT_ROOT = glue-factory/`` when they are not absolute — so the script
works from any working directory.

Decoders live in ``gluefactory.slam.rtabmap``; this script only reads the
DB tables and orchestrates the writes.

Examples:
    python -m gluefactory.scripts.prepare_map_dataset
    python -m gluefactory.scripts.prepare_map_dataset --config \\
        gluefactory/configs/map_dataset.yaml --clean --verbose
"""

import argparse
import logging
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gluefactory.slam.rtabmap import (
    RtabmapDB,
    camera_pose_from_node,
    decode_calibration,
    decode_depth,
    decode_image,
    decode_pose,
)

logging.basicConfig(level=logging.INFO, format="%(levelname).1s %(message)s")
logger = logging.getLogger("prepare_map_dataset")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Defaults (mirrored in gluefactory/configs/map_dataset.yaml)
# ---------------------------------------------------------------------------

DEFAULTS = {
    # Defaults are anchored to the workspace root (PROJECT_ROOT) so the
    # script works from any working directory.  Relative values in the
    # config file or CLI are also resolved against PROJECT_ROOT.
    "db": str(PROJECT_ROOT / "data/Map1/Map1.db"),
    "data_dir": str(PROJECT_ROOT / "data/Map1"),
    "keep_all": False,
    "write_depth": True,
    "write_calib": True,
    "clean": False,
    "verbose": False,
    "seed": 42,
    "frame": {
        "mode": "bins",                 # bins | minmax | all | stride
        "stride": 10,
        "min_trans": 0.02,              # meters (bins gate & minmax)
        "min_rot": 0.25,                # degrees (bins gate & minmax)
        "trans_bins": [0.02, 0.1, 0.25, 0.5, 1.0, 2.0],
        "rot_bins": [0, 2, 5, 10, 20, 45],
        "samples_per_bin": 300,
        "val_ratio": 0.17,
        "max_frames": 0,                # 0 = unlimited
    },
    "pairs": {
        "candidates": "links",          # links | spatial | all
        "link_types": [0, 1, 2],
        "spatial_radius": 0.75,
        "min_dist": 0.05,
        "max_dist": 2.0,
        "min_angle": 0.0,
        "max_angle": 45.0,
        "min_overlap": 0.0,             # 0 disables depth covisibility
        "trans_bins": [0.05, 0.25, 0.75, 1.5, 2.5],
        "rot_bins": [0, 3, 8, 15, 30, 45],
        "samples_per_bin": 120,
        "max_pairs_per_image": 200,
        "max_pairs": 0,                 # 0 = unlimited
        "val_ratio": 0.17,
    },
}


def deep_merge(base, overrides):
    out = dict(base)
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def build_parser():
    p = argparse.ArgumentParser(
        description="Build a SLAM dataset from an RTAB-Map .db",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--db", default=None)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--config", default=None, help="YAML config (overrides defaults)")
    p.add_argument("--keep_all", action="store_true", default=None,
                   help="extract every node (frames selection disabled)")
    p.add_argument("--no-depth", dest="write_depth", action="store_false", default=None)
    p.add_argument("--no-calib", dest="write_calib", action="store_false", default=None)
    p.add_argument("--clean", action="store_true", default=None, help="wipe data_dir first")
    p.add_argument("--seed", type=int, default=None)

    g = p.add_argument_group("frames")
    g.add_argument("--frame_mode", choices=["bins", "minmax", "all", "stride"], default=None)
    g.add_argument("--stride", type=int, default=None)
    g.add_argument("--min_trans", type=float, default=None, help="m")
    g.add_argument("--min_rot", type=float, default=None, help="deg")
    g.add_argument("--frame_trans_bins", type=float, nargs="+", default=None)
    g.add_argument("--frame_rot_bins", type=float, nargs="+", default=None)
    g.add_argument("--samples_per_bin_frames", type=int, default=None)
    g.add_argument("--frame_val_ratio", type=float, default=None)
    g.add_argument("--max_frames", type=int, default=None)

    p2 = p.add_argument_group("pairs")
    p2.add_argument("--candidates", choices=["links", "spatial", "all"], default=None)
    p2.add_argument("--link_types", type=int, nargs="+", default=None)
    p2.add_argument("--spatial_radius", type=float, default=None)
    p2.add_argument("--min_dist", type=float, default=None)
    p2.add_argument("--max_dist", type=float, default=None)
    p2.add_argument("--min_angle", type=float, default=None)
    p2.add_argument("--max_angle", type=float, default=None)
    p2.add_argument("--min_overlap", type=float, default=None)
    p2.add_argument("--trans_bins", type=float, nargs="+", default=None)
    p2.add_argument("--rot_bins", type=float, nargs="+", default=None)
    p2.add_argument("--samples_per_bin_pairs", type=int, default=None)
    p2.add_argument("--max_pairs_per_image", type=int, default=None)
    p2.add_argument("--max_pairs", type=int, default=None)
    p2.add_argument("--pair_val_ratio", type=float, default=None)
    return p


_FLAT_FRAME = {"frame_trans_bins": "trans_bins", "frame_rot_bins": "rot_bins",
               "samples_per_bin_frames": "samples_per_bin",
               "frame_val_ratio": "val_ratio", "frame_mode": "mode"}
_FLAT_PAIR = {"samples_per_bin_pairs": "samples_per_bin",
              "pair_val_ratio": "val_ratio"}


def load_config(args):
    conf = deep_merge(DEFAULTS, {})
    if args.config:
        path = Path(args.config)
        if path.exists():
            conf = deep_merge(conf, yaml.unsafe_load(path.read_text()))
        else:
            logger.warning(f"--config {args.config} not found, using defaults")
    flat = {}
    for k, v in vars(args).items():
        if v is None or k == "config":
            continue
        if k in _FLAT_FRAME:
            flat.setdefault("frame", {})[_FLAT_FRAME[k]] = v
        elif k in _FLAT_PAIR:
            flat.setdefault("pairs", {})[_FLAT_PAIR[k]] = v
        elif k in DEFAULTS["pairs"]:
            flat.setdefault("pairs", {})[k] = v
        elif k in DEFAULTS["frame"]:
            flat.setdefault("frame", {})[k] = v
        else:
            flat[k] = v
    return deep_merge(conf, flat)


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def to_hom(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def rel_delta(T_a, T_b):
    """Relative (translation_norm, rotation_deg) of frame a seen from b."""
    d = np.linalg.inv(T_b) @ T_a
    t = float(np.linalg.norm(d[:3, 3]))
    cos = np.clip((np.trace(d[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    ang = float(np.degrees(np.arccos(cos)))
    return t, ang


def bin_below(values, edges):
    """Return per-value bin index (>=1) or 0 when below ``edges[0]``.

    Values >= the last edge fall in the last bin.  Kept 1-based so 0 marks
    "below the min threshold".
    """
    return np.digitize(np.asarray(values, dtype=np.float64),
                       np.asarray(edges, dtype=np.float64))


def balanced_sample(rows, trans_edges, rot_edges, samples_per_bin, rng):
    """Sample *balanced* rows: up to ``samples_per_bin`` per (trans, rot) cell.

    Args:
        rows: (N, 4) float array ``[a, b, trans, ang]`` (any payload cols).
        trans_edges, rot_edges: bin edges (meters / degrees).
        samples_per_bin: cap per cell (0 = no cap).
        rng: numpy RandomState.

    Returns:
        np.array of row indices into ``rows`` (shuffled, balanced).
    """
    if rows.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    ti = bin_below(rows[:, 2], trans_edges)
    ri = bin_below(rows[:, 3], rot_edges)
    mask = (ti >= 1) & (ri >= 1)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return np.zeros(0, dtype=np.int64)
    groups = defaultdict(list)
    for r in idx.tolist():
        groups[(int(ti[r]), int(ri[r]))].append(r)
    chosen = []
    for grp in groups.values():
        arr = np.asarray(grp, dtype=np.int64)
        if samples_per_bin and arr.size > samples_per_bin:
            arr = arr[rng.permutation(arr.size)[:samples_per_bin]]
        chosen.append(arr)
    merged = np.concatenate(chosen)
    perm = rng.permutation(merged.size)
    return merged[perm]


def stem_for(stamp, node_id, used):
    s = f"{stamp:.9f}".replace(".", "_")
    if s not in used:
        return s
    k = 0
    while f"{s}_{k}" in used:
        k += 1
    return f"{s}_{k}"


# ---------------------------------------------------------------------------
# Frame + pair selection
# ---------------------------------------------------------------------------

def select_frames(node_ids, stamps, T, conf, rng):
    """Return the sorted list of node positions chosen as keyframes."""
    n = len(T)
    mode = conf["frame"]["mode"]
    mt = conf["frame"]["min_trans"]
    mr = conf["frame"]["min_rot"]

    if mode == "all":
        sel = np.arange(n)
    elif mode == "stride":
        sel = np.arange(0, n, max(1, conf["frame"]["stride"]))
    elif mode == "minmax":
        sel = [0]
        last = 0
        for k in range(1, n):
            t, a = rel_delta(T[k], T[last])
            if t >= mt or a >= mr:
                sel.append(k)
                last = k
        sel = np.asarray(sel, dtype=np.int64)
    elif mode == "bins":
        cand = np.zeros((n - 1, 4), dtype=np.float64)
        k = 0
        for a in range(n - 1):
            t, ang = rel_delta(T[a + 1], T[a])
            if t >= mt and ang >= mr:
                cand[k] = (a, a + 1, t, ang)
                k += 1
        cand = cand[:k]
        if k == 0:
            logger.warning("no transitions pass the min gates; keeping all nodes")
            return select_frames(node_ids, stamps, T, deep_merge(conf, {
                "frame": {"mode": "all"}}), rng)
        chosen = balanced_sample(cand, conf["frame"]["trans_bins"],
                                 conf["frame"]["rot_bins"],
                                 conf["frame"]["samples_per_bin"], rng)
        pos = set()
        for r in chosen.tolist():
            pos.add(int(cand[r, 0]))
            pos.add(int(cand[r, 1]))
        sel = np.asarray(sorted(pos), dtype=np.int64)
    else:
        raise ValueError(f"unknown frame mode {mode!r}")

    if conf["frame"]["max_frames"] and sel.size > conf["frame"]["max_frames"]:
        sel = np.sort(sel[rng.permutation(sel.size)[: conf["frame"]["max_frames"]]])
    if sel.size == 0:
        raise RuntimeError("frame selection is empty")
    return sel


def split_positions(positions, val_ratio, rng):
    n = len(positions)
    perm = rng.permutation(n)
    val_n = int(round(n * val_ratio))
    val = set(int(positions[i]) for i in perm[:val_n])
    train = set(int(positions[i]) for i in perm[val_n:])
    return train, val


def pair_candidates(db, positions, T, conf, rng):
    """Return deduped list of (pos_a, pos_b) geometric candidates."""
    from scipy.spatial import cKDTree

    pos = np.asarray(sorted(int(p) for p in positions), dtype=np.int64)
    pos_set = set(pos.tolist())
    mode = conf["pairs"]["candidates"]

    if mode == "links":
        cand = []
        seen = set()
        types = tuple(conf["pairs"]["link_types"])
        for a, b, _ltype in db.iter_links(types=types):
            if a not in pos_set or b not in pos_set:
                continue
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            cand.append((a, b))
        return cand

    trans = np.stack([T[p][:3, 3] for p in pos])
    tree = cKDTree(trans)
    radius = conf["pairs"]["max_dist"] if mode == "all" else conf["pairs"]["spatial_radius"]
    min_d = conf["pairs"]["min_dist"]
    cand = []
    seen = set()
    for i in range(len(pos)):
        neigh = tree.query_ball_point(trans[i], radius)
        for j in neigh:
            if j <= i:
                continue
            d = float(np.linalg.norm(trans[i] - trans[j]))
            if d < min_d:
                continue
            a, b = int(pos[i]), int(pos[j])
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            cand.append((a, b))
    return cand


def filter_and_balance_pairs(candidates, T, conf, rng):
    """Apply gates + per-bin balancing; return list of (a, b) pairs."""
    min_d = conf["pairs"]["min_dist"]
    max_d = conf["pairs"]["max_dist"]
    min_a = conf["pairs"]["min_angle"]
    max_a = conf["pairs"]["max_angle"]

    rows = []
    for a, b in candidates:
        t, ang = rel_delta(T[a], T[b])
        if t < min_d or t > max_d:
            continue
        if ang < min_a or ang > max_a:
            continue
        rows.append((a, b, t, ang))
    if not rows:
        return []
    rows = np.asarray(rows, dtype=np.float64)

    chosen = balanced_sample(rows, conf["pairs"]["trans_bins"],
                             conf["pairs"]["rot_bins"],
                             conf["pairs"]["samples_per_bin"], rng)
    sel = rows[chosen]

    if conf["pairs"]["max_pairs_per_image"]:
        cap = conf["pairs"]["max_pairs_per_image"]
        perm = rng.permutation(sel.shape[0])
        counts = defaultdict(int)
        kept = []
        for r in perm.tolist():
            a, b = int(sel[r, 0]), int(sel[r, 1])
            if counts[a] >= cap or counts[b] >= cap:
                continue
            counts[a] += 1
            counts[b] += 1
            kept.append(r)
        sel = sel[np.asarray(kept, dtype=np.int64)]

    if conf["pairs"]["max_pairs"] and sel.shape[0] > conf["pairs"]["max_pairs"]:
        sel = sel[rng.permutation(sel.shape[0])][: conf["pairs"]["max_pairs"]]

    return [(int(r[0]), int(r[1])) for r in sel]


# ---------------------------------------------------------------------------
# Extraction / IO
# ---------------------------------------------------------------------------

def _fmt_float(v):
    s = f"{v:.8g}"
    if "e" not in s and "i" not in s and "n" not in s and "." not in s:
        s += ".0"
    return s


def _write_calib_yaml(path, cal, stem):
    def mat(key, value, cols):
        value = np.asarray(value).reshape(-1)
        f.write(f"{key}:\n")
        f.write(f"  cols: {cols}\n")
        f.write(f"  rows: {value.size // cols}\n")
        f.write("  data:\n")
        for v in value:
            f.write(f"  - {_fmt_float(float(v))}\n")

    with open(path, "w") as f:
        f.write("%YAML:1.0\n")
        f.write("---\n")
        f.write(f"camera_name: {stem}\n")
        f.write(f"image_width: {int(cal['width'])}\n")
        f.write(f"image_height: {int(cal['height'])}\n")
        mat("camera_matrix", cal["K"], 3)
        mat("distortion_coefficients", cal["D"], 1)
        mat("rectification_matrix", cal["R_rect"], 3)
        mat("projection_matrix", cal["P"], 4)
        mat("local_transform", cal["local_transform"], 4)


def rot_to_quat(R):
    """Rotation matrix -> quaternion in TUM order ``(qx, qy, qz, qw)``.

    This is the order every consumer of ``poses_odom_RGBD_slam.txt`` expects
    (``gluefactory.datasets.slam_posed_images``,
    ``gluefactory.slam.geometry``), so the components must come back
    x, y, z, w.  The previous hand-rolled Shepperd implementation returned a
    different permutation depending on which branch it took -- (w, x, y, z)
    for ``tr > 0`` and for the ``m11``-dominant case, (-x, w, y, z) for the
    ``m00``-dominant one and (-w, y, z, x) for the fallback -- so the written
    poses did not reproduce the rotations stored in the DB.
    """
    return Rotation.from_matrix(np.asarray(R, dtype=np.float64)).as_quat()


def run(conf):
    data_dir = Path(conf["data_dir"])
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    db = Path(conf["db"])
    if not db.is_absolute():
        db = PROJECT_ROOT / db
    if conf["clean"]:
        logger.info(f"cleaning {data_dir}")
        shutil.rmtree(data_dir, ignore_errors=True)
    (data_dir / "images/rgb").mkdir(parents=True, exist_ok=True)
    (data_dir / "images/depth").mkdir(parents=True, exist_ok=True)
    (data_dir / "images/calib").mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(conf["seed"])

    node_ids, stamps, T = [], [], []
    with RtabmapDB(str(db)) as rdb:
        logger.info(f"loading nodes from {db} ...")
        for nid, stamp, pose in rdb.iter_nodes():
            R, t = decode_pose(pose)
            # Node pose is the base_link pose; compose the camera local
            # transform (base -> camera optical) to store the actual
            # camera-optical-frame pose in the world: T_w2cam = T_w2base @ T_base2cam
            _, _, cal_blob = rdb.node_data(nid)
            if cal_blob:
                cal = decode_calibration(cal_blob)
                R, t = camera_pose_from_node(R, t, cal["local_transform"])
            node_ids.append(nid)
            stamps.append(stamp)
            T.append(to_hom(R, t))
        node_ids = np.asarray(node_ids)
        stamps = np.asarray(stamps)
        T = [x.astype(np.float64) for x in T]
        logger.info(f"{len(node_ids)} nodes loaded (camera-optical poses)")

        if conf["keep_all"]:
            conf = deep_merge(conf, {"frame": {"mode": "all"}})
        positions = select_frames(node_ids, stamps, T, conf, rng)
        logger.info(f"selected {positions.size} keyframes "
                    f"(mode={conf['frame']['mode']})")

        frame_val_ratio = conf["frame"]["val_ratio"]
        if frame_val_ratio:
            train_set, val_set = split_positions(positions.tolist(), frame_val_ratio, rng)
        else:
            train_set, val_set = set(int(p) for p in positions), set()
        all_frames = sorted(train_set | val_set)
        logger.info(f"frames: train={len(train_set)} val={len(val_set)}")

        candidates = pair_candidates(rdb, all_frames, T, conf, rng)
        logger.info(f"{len(candidates)} candidate pairs "
                    f"(mode={conf['pairs']['candidates']})")

        def keep(split_set):
            return [(a, b) for (a, b) in candidates if a in split_set and b in split_set]

        train_pairs = filter_and_balance_pairs(keep(train_set), T, conf, rng)
        val_pairs = filter_and_balance_pairs(keep(val_set), T, conf, rng)
        logger.info(f"pairs: train={len(train_pairs)} val={len(val_pairs)}")

        pair_frames = set()
        for a, b in train_pairs + val_pairs:
            pair_frames.add(a)
            pair_frames.add(b)
        frames_out = sorted(set(all_frames) | pair_frames)

        stem_of = {}
        id_of = {}
        used = set()
        for p in frames_out:
            s = stem_for(stamps[p], int(node_ids[p]), used)
            used.add(s)
            stem_of[p] = s
            id_of[p] = int(node_ids[p])

        with open(data_dir / "poses_odom_RGBD_slam.txt", "w") as f:
            f.write("#timestamp x y z qx qy qz qw id\n")
            for p in frames_out:
                q = rot_to_quat(T[p][:3, :3])
                t = T[p][:3, 3]
                f.write(f"{stem_of[p]} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                        f"{q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f} {id_of[p]}\n")

        def write_list(path, frames):
            with open(path, "w") as f:
                for p in frames:
                    f.write(f"rgb/{stem_of[p]}.png\n")

        write_list(data_dir / "image_list_train.txt",
                   [p for p in frames_out if p in train_set])
        write_list(data_dir / "image_list_val.txt",
                   [p for p in frames_out if p in val_set])

        def write_pairs(path, pairs):
            with open(path, "w") as f:
                for a, b in pairs:
                    f.write(f"rgb/{stem_of[a]}.png rgb/{stem_of[b]}.png\n")

        write_pairs(data_dir / "pairs_train.txt", train_pairs)
        write_pairs(data_dir / "pairs_val.txt", val_pairs)
        logger.info("wrote poses + image lists + pair lists")

        ids_out = set(id_of[p] for p in frames_out)
        n_img = n_depth = n_cal = 0
        for nid, _stamp, _pose in rdb.iter_nodes():
            if int(nid) not in ids_out:
                continue
            p = next(pp for pp in frames_out if id_of[pp] == int(nid))
            stem = stem_of[p]
            img_blob, dep_blob, cal_blob = rdb.node_data(nid)
            img = decode_image(img_blob)
            if img is not None:
                cv2.imwrite(str(data_dir / "images/rgb" / f"{stem}.png"), img)
                n_img += 1
            if conf["write_depth"] and dep_blob:
                dep = decode_depth(dep_blob)
                if dep is not None:
                    cv2.imwrite(str(data_dir / "images/depth" / f"{stem}.png"), dep)
                    n_depth += 1
            if conf["write_calib"] and cal_blob:
                cal = decode_calibration(cal_blob)
                _write_calib_yaml(data_dir / "images/calib" / f"{stem}.yaml", cal, stem)
                n_cal += 1

        logger.info(f"extracted {n_img} rgb, {n_depth} depth, {n_cal} calib "
                    f"-> {data_dir}")


def main():
    args = build_parser().parse_args()
    conf = load_config(args)
    run(conf)


if __name__ == "__main__":
    main()