"""Read-only access to an RTAB-Map SQLite database (``.db``) and decoders
for the binary blobs stored in it.

Verified against a real 8683-node map:

* ``Node``      — every row is a keyframe (its local graph ``Map `` row has
  an image).  ``pose`` is a 3x4 float32 = [R|t], T_wc, orthonormal det=+1.
* ``Link``      — graph edges; ``type`` 0 = neighbor, 1 = global closure,
  2 = local closure, 9 = self-prior (ignore).
* ``Data``      — ``image`` is a JPEG blob (0xFFD8…), ``depth`` is an RVL
  blob starting with ``DEPTHRVL``, ``calibration`` is a 372-byte packed
  OpenCV camera-model file.
* ``Calibration_``/``LocalTransform``/``CameraMeta_`` — per-camera metadata
  (we read the packed ``calibration`` blob directly, same as RTAB-Map does).

Depth units are 16-bit millimeters (0 = invalid); the RVL codec is ported
in :mod:`gluefactory.slam.rvl`.
"""

import sqlite3
from pathlib import Path

import cv2
import numpy as np

from .rvl import decompress_depth

__all__ = [
    "RtabmapDB",
    "decode_pose",
    "decode_image",
    "decode_depth",
    "decode_calibration",
    "pose_matrix",
    "camera_pose_from_node",
    "CALIB_SIZEOF",
]

# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------

_CALIB_DTYPE_DEF = np.dtype(
    [
        ("header", np.int32, (11,)),
        ("K", np.float64, (9,)),
        ("D", np.float64, (5,)),
        ("R_rect", np.float64, (9,)),
        ("P", np.float64, (12,)),
        ("local_transform", np.float32, (12,)),
    ]
)
CALIB_SIZEOF = _CALIB_DTYPE_DEF.itemsize  # 372 bytes


def decode_pose(pose_bytes):
    """Decode the 3x4 float32 ``[R | t]`` pose stored on a Node row.

    The Node pose is the robot **base_link** pose in the map frame
    (``T_map2base``), *not* the camera pose.
    """
    mat = np.frombuffer(pose_bytes, dtype=np.float32).reshape(3, 4)
    R = mat[:, :3].copy()
    t = mat[:, 3].copy()
    return R, t


def pose_matrix(R, t):
    """Make a 4x4 homogeneous matrix from a 3x3 rotation and translation."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


def camera_pose_from_node(R_base, t_base, local_transform):
    """Compose a Node (base) pose with the camera local transform.

    RTAB-Map stores ``Node.pose`` = pose of the robot base in the map and
    the camera optical frame relative to the base in the calibration blob's
    ``local_transform`` (= ``T_base2cam``).  The camera pose in the world is
    ``T_map2cam = T_node @ T_base2cam`` (same composition as RTAB-Map's
    Export tool: ``cameraViewpoint = nodePose * localTransform``).

    Args:
        R_base, t_base: rotation/translation of the Node pose (map -> base).
        local_transform: (3, 4) ``[R | t]`` from the decoded calibration.

    Returns:
        (R_cam [3,3], t_cam [3]) of the camera optical frame in the map.
    """
    lt = np.asarray(local_transform, dtype=np.float64).reshape(3, 4)
    T_wc = pose_matrix(R_base, t_base) @ pose_matrix(lt[:, :3], lt[:, 3])
    return T_wc[:3, :3].copy(), T_wc[:3, 3].copy()


def decode_image(image_blob):
    """Decode the JPEG (or any imdecode-able) image blob; BGR or gray."""
    if not image_blob:
        return None
    img = cv2.imdecode(np.frombuffer(image_blob, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img


def decode_depth(depth_blob):
    """Decode a DEPTHRVL blob into a uint16 (mm) depth map.

    Assumes the blob starts with the 16-byte header already described in
    ``Compression.cpp``; no validation is performed on the magic.
    """
    if not depth_blob or len(depth_blob) < 16:
        return None
    width = int(np.frombuffer(depth_blob[8:12], dtype="<u4")[0])
    height = int(np.frombuffer(depth_blob[12:16], dtype="<u4")[0])
    payload = depth_blob[16:]
    depth = decompress_depth(payload, height * width)
    return depth.reshape(height, width)


def decode_calibration(calibration_blob):
    """Decode the 372-byte packed OpenCV calibration into a dict.

    Offsets (double-checked against funcs.cpp and an existing MAP2 calib
    yaml): 44-byte int32 header (width=+16, height=+20), then K (9 f64),
    D (5 f64), R_rect (9 f64), P (12 f64), local_transform (12 f32).
    """
    raw = np.frombuffer(calibration_blob, dtype=np.uint8)
    # Align: the packed byte count must match the struct.
    assert raw.size % CALIB_SIZEOF == 0, f"bad calibration blob: {raw.size} bytes"
    cal = raw.view(_CALIB_DTYPE_DEF)[0]
    header = cal["header"]
    R = np.array(cal["R_rect"], dtype=np.float64).reshape(3, 3)
    K = np.array(cal["K"], dtype=np.float64).reshape(3, 3)
    D = np.array(cal["D"], dtype=np.float64)
    P = np.array(cal["P"], dtype=np.float64).reshape(3, 4)
    local_transform = np.array(cal["local_transform"], dtype=np.float32).reshape(3, 4)
    return {
        "header": np.asarray(header),
        "width": int(header[4]),
        "height": int(header[5]),
        "K": K,
        "D": D,
        "R_rect": R,
        "P": P,
        "local_transform": local_transform,
    }


# ---------------------------------------------------------------------------
# Lightweight read-only DB access
# ---------------------------------------------------------------------------

class RtabmapDB:
    """Read-only sqlite3 wrapper over an RTAB-Map database file."""

    def __init__(self, db_path):
        self.path = Path(db_path)
        uri = "file:{}?mode=ro".format(self.path.resolve())
        self.con = sqlite3.connect(uri, uri=True)
        self.con.row_factory = sqlite3.Row

    def close(self):
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- Node / pose --------------------------------------------------------

    def iter_nodes(self, map_id=0):
        """Yield (node_id, stamp, pose_bytes) for every node in a map."""
        cur = self.con.execute(
            "SELECT id, stamp, pose FROM Node WHERE map_id=? ORDER BY id", (map_id,)
        )
        for row in cur:
            yield row["id"], float(row["stamp"]), row["pose"]

    def node_count(self, map_id=0):
        cur = self.con.execute("SELECT COUNT(*) FROM Node WHERE map_id=?", (map_id,))
        return cur.fetchone()[0]

    # -- Data ---------------------------------------------------------------

    def node_data(self, node_id):
        """Return (image_blob, depth_blob, calibration_blob) for a node."""
        row = self.con.execute(
            "SELECT image, depth, calibration FROM Data WHERE id=?", (node_id,)
        ).fetchone()
        if row is None:
            return None, None, None
        return row["image"], row["depth"], row["calibration"]

    def iter_data(self):
        cur = self.con.execute("SELECT id, image, depth, calibration FROM Data ORDER BY id")
        for row in cur:
            yield row["id"], row["image"], row["depth"], row["calibration"]

    # -- Link ---------------------------------------------------------------

    def iter_links(self, types=(0, 1, 2)):
        """Yield (from_id, to_id, type) for links whose type is in ``types``."""
        if not types:
            return
        placeholders = ",".join("?" * len(types))
        cur = self.con.execute(
            f"SELECT from_id, to_id, type FROM Link WHERE type IN ({placeholders}) "
            "ORDER BY from_id, to_id",
            tuple(int(t) for t in types),
        )
        for row in cur:
            yield row["from_id"], row["to_id"], int(row["type"])