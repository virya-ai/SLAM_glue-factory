"""Python/numpy port of the RTAB-Map ``RvlCodec`` lossless depth codec.

The reference implementation is a C++ wrapper of the algorithm presented by
Andrew D. Wilson in "Fast Lossless Depth Image Compression" (SIGCHI'17),
distributed under the MIT license (see ``rtabmap/corelib/src/rvl_codec.cpp``
and ``rtabmap/corelib/include/rtabmap/core/rvl_codec.h`` in the RTAB-Map
source tree).  This module re-implements the decompressor in vectorized
numpy so a ``.db`` depth blob can be decoded without the RTAB-Map C++
bindings.

Depth blob layout (see ``corelib/src/Compression.cpp``)::

    bytes[0:8]    b"DEPTHRVL"          magic signature
    bytes[8:12]   uint32 LE  cols (= image width)
    bytes[12:16]  uint32 LE  rows (= image height)
    bytes[16:]    RVL bitstream (32-bit words, big-endian nibble order)

Bits above the top are None.  No file is written here; only in-memory
numpy arrays are produced.
"""

import numpy as np

__all__ = ["compress_depth", "decompress_depth"]


def pipelined_decompress_depth(payload, height, width):
    """Alias kept for readability — see :func:`decompress_depth`."""
    return decompress_depth(payload, height * width).reshape(height, width)


def _decode_values(payload):
    """Decode the VLE token stream of an RVL payload into an integer vector.

    Every VLE value is read most-significant nibble first and reassembled
    as ``g0 | g1 << 3 | g2 << 6 | ...`` — identical to the reference
    ``DecodeVLE()``.
    """
    if not payload:
        return np.zeros(0, dtype=np.uint64)
    words = np.frombuffer(payload, dtype="<u4").astype(np.uint64)
    if words.size == 0:
        return np.zeros(0, dtype=np.uint64)

    shifts = np.tile(np.arange(28, -4, -4), words.size).astype(np.uint8)
    nibbles = ((np.repeat(words, 8) >> shifts) & 0x0F).astype(np.uint64)
    cont = (nibbles >> 3) & 1  # 1 = more nibbles follow for this value
    pay = nibbles & 0x07

    ends = np.flatnonzero(cont == 0)
    if ends.size == 0:
        return np.zeros(0, dtype=np.uint64)

    starts = np.empty(ends.size, dtype=np.int64)
    starts[0] = 0
    starts[1:] = ends[:-1] + 1

    lengths = (ends - starts + 1).astype(np.int64)
    reps = np.repeat(starts, lengths)
    group_idx = np.arange(nibbles.size, dtype=np.int64) - reps

    segment = (pay << (3 * group_idx.astype(np.uint8)))
    values = np.add.reduceat(segment, starts.astype(np.intp))
    return values


def decompress_depth(payload, num_pixels):
    """Decompress an RVL payload into ``uint16`` depth values.

    Args:
        payload:    raw compressed bytes (after the 16-byte header).
        num_pixels: ``height * width`` of the uncompressed depth image.

    Returns:
        np.ndarray shape (num_pixels,) dtype uint16.  Zero means invalid /
        no-depth pixel, everything else is depth in mm (16-bit units).
    """
    values = _decode_values(payload)
    out = np.zeros(num_pixels, dtype=np.uint16)
    n = values.shape[0]
    t = 0
    p = 0
    prev = 0
    while p < num_pixels and t + 1 < n:
        zeros = int(values[t])
        nonzeros = int(values[t + 1])
        t += 2

        p += zeros
        if p >= num_pixels:
            break
        if nonzeros <= 0:
            continue

        max_take = nonzeros
        if t + max_take > n:
            max_take = n - t
        if max_take <= 0:
            break
        seg = values[t:t + max_take]
        t += max_take

        s = seg.astype(np.int64)
        delta = np.where((seg & 1) == 1, -((s >> 1) + 1), s >> 1)

        seg_vals = (np.cumsum(delta, dtype=np.int64) + prev) & 0xFFFF

        end = p + max_take
        if end > num_pixels:
            end = num_pixels
        out[p:end] = seg_vals[: end - p].astype(np.uint16)
        prev = int(seg_vals[max_take - 1])
        p = end
    return out


# ---------------------------------------------------------------------------
# Reference encoder (used only for round-trip tests)
# ---------------------------------------------------------------------------

class _BitWriter:
    """Mirror of the C++ ``EncodeVLE`` word packing."""

    def __init__(self):
        self.words = []
        self.word = 0
        self.nibbles_written = 0

    def write_vle(self, value):
        while True:
            nibble = value & 0x7
            value >>= 3
            if value:
                nibble |= 0x8
            self.word = (self.word << 4) | nibble
            self.nibbles_written += 1
            if self.nibbles_written == 8:
                self.words.append(self.word)
                self.word = 0
                self.nibbles_written = 0
            if not value:
                break

    def finish(self):
        if self.nibbles_written:
            self.words.append(self.word << (4 * (8 - self.nibbles_written)))
            self.word = 0
            self.nibbles_written = 0
        return b"".join(
            int(w).to_bytes(4, "little", signed=False) for w in self.words
        )


def compress_depth(depth):
    """Reference compressor for testing (mirrors ``CompressRVL``).

    Args:
        depth: np.ndarray uint16 anything flattenable.

    Returns:
        bytes payload (without the DEPTHRVL header).
    """
    seq = np.asarray(depth, dtype=np.uint16).reshape(-1)
    writer = _BitWriter()
    i = 0
    n = seq.size
    previous = 0
    while i < n:
        zeros = 0
        while i < n and seq[i] == 0:
            zeros += 1
            i += 1
        writer.write_vle(zeros)
        j = i
        nonzeros = 0
        while j < n and seq[j] != 0:
            nonzeros += 1
            j += 1
        writer.write_vle(nonzeros)
        for _ in range(nonzeros):
            current = int(seq[i])
            delta = current - previous
            positive = (delta << 1) ^ (delta >> 31)
            writer.write_vle(positive)
            previous = current
            i += 1
    return writer.finish()


def roundtrip(depth):
    """Encode + decode a depth map; returns decoded uint16 array (flattened)."""
    from .rvl import decompress_depth

    payload = compress_depth(depth)
    return decompress_depth(payload, int(np.asarray(depth).size))