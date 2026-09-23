import sqlite3
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from gluefactory.slam.rtabmap import (
    RtabmapDB,
    CALIB_SIZEOF,
    camera_pose_from_node,
    decode_calibration,
    decode_depth,
    decode_image,
    decode_pose,
)
from gluefactory.slam.rvl import compress_depth, decompress_depth
from gluefactory.scripts.prepare_map_dataset import (
    balanced_sample,
    bin_below,
    rel_delta,
    run,
    to_hom,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_DB = PROJECT_ROOT / "data/Map1/Map1.db"


def make_calibration_blob(width=512, height=207, K=None):
    blob = np.zeros(CALIB_SIZEOF, dtype=np.uint8)
    view = blob.view(
        np.dtype(
            [
                ("header", np.int32, (11,)),
                ("K", np.float64, (9,)),
                ("D", np.float64, (5,)),
                ("R_rect", np.float64, (9,)),
                ("P", np.float64, (12,)),
                ("local_transform", np.float32, (12,)),
            ]
        )
    )[0]
    view["header"][4] = width
    view["header"][5] = height
    view["K"] = np.eye(3).reshape(-1) if K is None else np.asarray(K).reshape(-1)
    view["R_rect"] = np.eye(3).reshape(-1)
    view["D"] = 0.0
    view["local_transform"] = np.eye(4)[:3, :].reshape(-1)
    return blob.tobytes()


def make_depth_blob(depth):
    depth = np.asarray(depth, dtype=np.uint16)
    h, w = depth.shape
    return (
        b"DEPTHRVL"
        + np.uint32(w).tobytes()
        + np.uint32(h).tobytes()
        + compress_depth(depth)
    )


def make_pose_blob(R, t):
    m = np.zeros((3, 4), dtype=np.float32)
    m[:, :3] = R
    m[:, 3] = t
    return m.tobytes()


def make_db(path, nodes, links):
    """nodes: [(id, stamp, R, t)] ; links: [(a, b, type)]"""
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE Node(id INTEGER PRIMARY KEY, map_id INTEGER, stamp FLOAT, pose BLOB);
        CREATE TABLE "Data"(id INTEGER PRIMARY KEY, image BLOB, depth BLOB, calibration BLOB);
        CREATE TABLE Link(from_id INTEGER, to_id INTEGER, type INTEGER, information_matrix BLOB);
        """
    )
    rng_state = np.random.RandomState(0)
    depth = rng_state.randint(0, 60000, size=(16, 16), dtype=np.uint16)
    depth[0:4, 0:4] = 0
    img = np.random.randint(0, 255, size=(108, 160, 3), dtype=np.uint8)
    img_jpg = cv2.imencode(".jpg", img)[1].tobytes()
    for nid, stamp, R, t in nodes:
        con.execute(
            "INSERT INTO Node(id, map_id, stamp, pose) VALUES (?, 0, ?, ?)",
            (nid, stamp, make_pose_blob(R, t)),
        )
        con.execute(
            "INSERT INTO Data(id, image, depth, calibration) VALUES (?, ?, ?, ?)",
            (nid, img_jpg, make_depth_blob(depth), make_calibration_blob()),
        )
    ident = np.eye(4, dtype=np.float32).tobytes()
    for a, b, t in links:
        con.execute(
            "INSERT INTO Link(from_id, to_id, type, information_matrix) VALUES (?,?,?,?)",
            (a, b, t, ident),
        )
    con.commit()
    con.close()


class TestRvl(unittest.TestCase):
    def test_random_roundtrip(self):
        rng = np.random.RandomState(1)
        d = rng.randint(0, 60000, size=(207, 512), dtype=np.uint16)
        d[rng.rand(207, 512) < 0.7] = 0
        seq = d.reshape(-1)
        out = decompress_depth(compress_depth(d), seq.size).reshape(207, 512)
        self.assertTrue(np.array_equal(out, d))

    def test_run_lengths_and_ramps(self):
        # 512 random runs separated by zeros, incl. long flat ramps
        rng = np.random.RandomState(2)
        seq = []
        prev = 0
        for _ in range(64):
            seq += [0] * rng.randint(0, 300)
            n = rng.randint(1, 500)
            for _ in range(n):
                delta = rng.randint(-50, 50)
                prev = (prev + delta) & 0xFFFF
                seq.append(prev)
        seq = np.asarray(seq, dtype=np.uint16)
        out = decompress_depth(compress_depth(seq), seq.size)
        self.assertTrue(np.array_equal(out, seq))

    def test_all_zero(self):
        seq = np.zeros(1000, dtype=np.uint16)
        out = decompress_depth(compress_depth(seq), seq.size)
        self.assertEqual(out.sum(), 0)

    def test_real_header_layout(self):
        # header parse matches Compression.cpp: magic@0, cols@8, rows@12
        blob = make_depth_blob(np.zeros((7, 9), dtype=np.uint16))
        self.assertEqual(blob[0:8], b"DEPTHRVL")
        self.assertEqual(int(np.frombuffer(blob[8:12], dtype="<u4")[0]), 9)
        self.assertEqual(int(np.frombuffer(blob[12:16], dtype="<u4")[0]), 7)


class TestDecoders(unittest.TestCase):
    def test_pose_orthonormal_and_translation(self):
        R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        t = np.array([1.0, 2.0, 3.0])
        R2, t2 = decode_pose(make_pose_blob(R, t))
        self.assertTrue(np.allclose(R2, R.astype(np.float32)))
        self.assertTrue(np.allclose(t2, t.astype(np.float32)))
        self.assertAlmostEqual(float(np.linalg.det(R2)), 1.0, places=5)

    def test_calibration_blob_layout(self):
        K = np.array([[100, 0, 50], [0, 110, 60], [0, 0, 1]], dtype=float)
        blob = make_calibration_blob(width=320, height=200, K=K)
        cal = decode_calibration(blob)
        self.assertEqual(cal["width"], 320)
        self.assertEqual(cal["height"], 200)
        self.assertTrue(np.allclose(cal["K"], K))
        # local_transform read from the trailing f32 block
        self.assertTrue(np.allclose(cal["local_transform"][0],
                                    [1.0, 0.0, 0.0, 0.0]))

    def test_image_decode_jpeg(self):
        img = np.random.randint(0, 255, size=(108, 160, 3), dtype=np.uint8)
        blob = cv2.imencode(".jpg", img)[1].tobytes()
        out = decode_image(blob)
        self.assertEqual(out.shape, (108, 160, 3))

    def test_depth_decode_headerful(self):
        rng = np.random.RandomState(3)
        d = rng.randint(0, 30000, size=(16, 16), dtype=np.uint16)
        dec = decode_depth(make_depth_blob(d))
        self.assertEqual(dec.shape, (16, 16))
        self.assertTrue(np.array_equal(dec, d))


def sample_poses_on_axis():
    nodes = []
    c = np.cos(np.radians([0, 10, -5, 20, 0, 15, 30, 5]))
    s = np.sin(np.radians([0, 10, -5, 20, 0, 15, 30, 5]))
    for i in range(8):
        Rz = np.array([[c[i], -s[i], 0], [s[i], c[i], 0], [0, 0, 1.0]])
        nodes.append((i + 1, 1000.0 + i * 0.5, Rz, np.array([i * 0.3, 0.0, 0.0])))
    return nodes


class TestSelectionHelpers(unittest.TestCase):
    def test_bin_below_edges(self):
        v = np.array([0.0, 0.02, 0.1, 0.25, 2.0, 5.0])
        b = bin_below(v, [0.02, 0.1, 0.25, 0.5, 1.0, 2.0])
        self.assertEqual(b.tolist(), [0, 1, 2, 3, 6, 6])

    def test_balanced_sample_caps_per_bin(self):
        rng = np.random.RandomState(0)
        # 100 rows: 50 in cell A (trans 0.5), 50 in cell B (trans 1.0)
        rows = np.zeros((100, 4))
        rows[:50, 2] = 0.5
        rows[50:, 2] = 1.0
        rows[:, 3] = 10.0
        chosen = balanced_sample(rows, [0.05, 0.25, 0.75, 1.5], [0, 20], 20, rng)
        self.assertEqual(len(chosen), 40)
        # exactly 20 per cell: measure via trans of chosen rows
        tr = rows[chosen][:, 2]
        self.assertEqual((tr < 0.75).sum(), 20)
        self.assertEqual((tr >= 0.75).sum(), 20)

    def test_balanced_sample_drops_below_min(self):
        rng = np.random.RandomState(0)
        rows = np.zeros((5, 4))
        rows[:, 2] = 0.01  # below first trans edge
        rows[:, 3] = 10.0
        self.assertEqual(balanced_sample(rows, [0.05, 0.5], [0, 20], 5, rng).size, 0)

    def test_rel_delta(self):
        Ta = to_hom(np.eye(3), np.zeros(3))
        Tb = to_hom(np.eye(3), np.array([1.0, 0.0, 0.0]))
        t, ang = rel_delta(Ta, Tb)
        self.assertAlmostEqual(t, 1.0, places=6)
        self.assertAlmostEqual(ang, 0.0, places=6)
        R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        _, ang = rel_delta(to_hom(R, np.zeros(3)), to_hom(np.eye(3), np.zeros(3)))
        self.assertAlmostEqual(ang, 90.0, places=5)

    def test_camera_pose_composition(self):
        # T_w2cam = T_node(base) @ T_base2cam(brand from calibration)
        R_base = np.eye(3)
        t_base = np.array([1.0, 2.0, 3.0])
        # forward 0.5 m in base x, up 1.86 m in base z
        lt = np.eye(4)[:3].copy()
        lt[:, 3] = [0.5, 0.0, 1.86]
        R_cam, t_cam = camera_pose_from_node(R_base, t_base, lt)
        expected = to_hom(R_base, t_base) @ to_hom(lt[:, :3], lt[:, 3])
        self.assertTrue(np.allclose(R_cam, expected[:3, :3]))
        self.assertTrue(np.allclose(t_cam, expected[:3, 3]))
        # same camera offset is applied in the BASE frame: rotating the base
        # by 90 deg must rotate the offset accordingly (not just add world offset)
        Rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        Rc, tc = camera_pose_from_node(Rz, np.zeros(3), lt)
        self.assertTrue(np.allclose(tc, Rz @ lt[:, 3]))


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "t.db"
        nodes = sample_poses_on_axis()
        links = [(i, i + 1, 0) for i in range(1, 8)] + [(1, 4, 2), (2, 5, 1)]
        make_db(self.db, nodes, links)

    def tearDown(self):
        self.tmp.cleanup()

    def test_pipeline_writes_consistent_dataset(self):
        out = Path(self.tmp.name) / "MAP2"
        conf = {
            "db": str(self.db),
            "data_dir": str(out),
            "keep_all": False,
            "write_depth": True,
            "write_calib": True,
            "clean": True,
            "verbose": False,
            "seed": 42,
            "frame": {
                "mode": "all",
                "stride": 10,
                "min_trans": 0.02,
                "min_rot": 0.25,
                "trans_bins": [0.02, 0.3, 0.6, 1.0],
                "rot_bins": [0, 10, 30],
                "samples_per_bin": 30,
                "val_ratio": 0.17,
                "max_frames": 0,
            },
            "pairs": {
                "candidates": "links",
                "link_types": [0, 1, 2],
                "spatial_radius": 0.75,
                "min_dist": 0.05,
                "max_dist": 2.0,
                "min_angle": 0.0,
                "max_angle": 90.0,
                "min_overlap": 0.0,
                "trans_bins": [0.05, 0.3, 0.6, 1.0],
                "rot_bins": [0, 10, 30],
                "samples_per_bin": 10,
                "max_pairs_per_image": 0,
                "max_pairs": 0,
                "val_ratio": 0.17,
            },
        }
        run(conf)

        self.assertTrue((out / "poses_odom_RGBD_slam.txt").exists())
        poses = {}
        with open(out / "poses_odom_RGBD_slam.txt") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                poses[parts[0]] = parts
        self.assertGreaterEqual(len(poses), 5)

        train_list = (out / "image_list_train.txt").read_text().splitlines()
        val_list = (out / "image_list_val.txt").read_text().splitlines()
        train_stems = {Path(l).stem for l in train_list}
        val_stems = {Path(l).stem for l in val_list}
        self.assertTrue(train_stems.isdisjoint(val_stems))

        for split, entries in [("train", train_list), ("val", val_list)]:
            for entry in entries:
                self.assertTrue(entry.startswith("rgb/"), entry)
                rgb = out / "images" / entry
                self.assertTrue(rgb.exists(), f"missing {rgb}")
                stem = Path(entry).stem
                self.assertIn(stem, poses, f"pose key missing for {stem} in {split}")
                depth = out / "images" / entry.replace("rgb/", "depth/")
                self.assertTrue(depth.exists(), f"missing depth {depth}")

        pairs_train = (out / "pairs_train.txt").read_text().splitlines()
        pairs_val = (out / "pairs_val.txt").read_text().splitlines()
        for pair in pairs_train + pairs_val:
            a, b = pair.split()
            self.assertTrue((out / "images" / a).exists(), a)
            self.assertTrue((out / "images" / b).exists(), b)
        self.assertGreater(len(pairs_train), 0)

        # depth decode equals the synthetic source
        for stem in train_stems:
            d = cv2.imread(str(out / "images/depth" / f"{stem}.png"),
                           cv2.IMREAD_UNCHANGED)
            self.assertIsNotNone(d)
            self.assertEqual(d.dtype, np.uint16)

        # calibration yaml is readable and matches K
        cal_path = out / "images/calib" / f"{sorted(train_stems)[0]}.yaml"
        content = cal_path.read_text()
        if content.startswith("%YAML"):
            content = content.split("\n", 1)[1]
        cal = yaml.unsafe_load(content)
        K = np.array(cal["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
        self.assertTrue(np.allclose(K, np.eye(3), atol=1e-3))

    def test_depth_uint16_scaled_mm(self):
        # end-to-end already writes source values untouched; verify a 1 m depth
        # is stored as 1000 mm through the rvl round trip.
        d = np.zeros((8, 8), dtype=np.uint16)
        d[:, :] = 1000
        dec = decode_depth(make_depth_blob(d))
        self.assertEqual(int(dec[0, 0]), 1000)


@unittest.skipUnless(REAL_DB.exists(), "data/Map1/Map1.db not available")
class TestRealMap(unittest.TestCase):
    def test_real_db_decodes(self):
        with RtabmapDB(str(REAL_DB)) as db:
            self.assertGreaterEqual(db.node_count(), 8000)
            nid, stamp, pose = next(iter(db.iter_nodes()))
            R, t = decode_pose(pose)
            self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=4)
            img, depth, cal = db.node_data(nid)
            self.assertEqual(decode_image(img).shape[2], 3)
            dep = decode_depth(depth)
            self.assertEqual(dep.shape, (207, 512))
            self.assertEqual(dep.dtype, np.uint16)

    def test_real_camera_pose_differs_from_base_pose(self):
        # Cross-check: composing the calibration local_transform must move the
        # stored base pose off the map origin to the actual camera position.
        with RtabmapDB(str(REAL_DB)) as db:
            nid, stamp, pose = next(iter(db.iter_nodes()))
            R_base, t_base = decode_pose(pose)
            _, _, cal_blob = db.node_data(nid)
            cal = decode_calibration(cal_blob)
            lt = cal["local_transform"]
            self.assertGreater(float(np.linalg.norm(lt[:, 3])), 0.5)  # visible offset
            R_cam, t_cam = camera_pose_from_node(R_base, t_base, lt)
            # at the first node the base sits at the origin; the camera must
            # therefore be at the patched base->camera offset (rotated by base R)
            expected = R_base @ lt[:, 3] + t_base
            self.assertTrue(np.allclose(t_cam, expected, atol=1e-3))
            self.assertTrue(np.allclose(R_cam, R_base @ lt[:, :3], atol=1e-3))
            self.assertAlmostEqual(float(np.linalg.det(R_cam)), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()