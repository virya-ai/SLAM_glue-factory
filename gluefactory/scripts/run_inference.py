"""
Run SuperPoint alone, or SuperPoint + SuperGlue/LightGlue, on images and
visualize the result.

Two backends:
  --backend checkpoint   Load .tar training checkpoints via
                         gluefactory.slam.matcher.SLAMMatcher.
  --backend exported     Load .pt TorchScript graphs produced by
                         gluefactory.scripts.trace_model, decoding
                         SuperPoint's dense heatmap with the same NMS/
                         threshold/top-k logic used at training time
                         (gluefactory.models.extractors.superpoint_open).

Three matcher choices (any of the extractor's outputs can be visualized
alone by passing --matcher none):
  --matcher none | superglue | lightglue

Output modes, combinable (repeated or space-separated values):
  html  interactive canvas-overlay dashboard (pairs) or keypoint grid (--matcher none)
  data  matches_data.js only (no HTML)
  png   static per-pair or per-image PNGs
  all   all of the above

For 5k+ batches: use "--output html data" (skip the per-pair matplotlib
plots — the main wall-time sink) and/or add png with --save_workers 8.

ONNX backend (recommended deployment path):
  The .onnx files are hardware-neutral. onnxruntime picks the accelerator
  at load time via --execution_provider, so the SAME model files run on
  CPU, NVIDIA GPUs, or Intel GPUs:
    cpu     CPUExecutionProvider        (pip install onnxruntime)
    cuda    CUDAExecutionProvider       (pip install onnxruntime-gpu)
    openvino OpenVINOExecutionProvider  (Intel iGPU/Arc, device_type=GPU;
                                         pip install onnxruntime-openvino)
    auto    first available of the above
  Batch directory inference runs chunk-wise on the accelerator -- every
  chunk of images through the extractor in one batched forward; LightGlue
  stays per-pair (batch=1 by design), see --batch_size.

Usage:
    # SuperPoint + LightGlue, from training checkpoints, full dashboard
    MPLBACKEND=Agg python -m gluefactory.scripts.run_inference \\
        --backend checkpoint --matcher lightglue \\
        --extractor_ckpt outputs/training/superpoint_slam_run/checkpoint_best.tar \\
        --matcher_ckpt   outputs/training/lightglue_slam_run/checkpoint_best.tar \\
        --input data/output/slam/images/rgb \\
        --output all --output_dir data/output/slam/visualizations/match_inference_rgb_light

    # SuperPoint alone, from an exported .pt, static PNGs only
    MPLBACKEND=Agg python -m gluefactory.scripts.run_inference \\
        --backend exported --matcher none \\
        --extractor_pt superpoint.pt \\
        --input data/output/slam/images/rgb \\
        --output png --output_dir outputs/sp_detections

    # SuperPoint + LightGlue, single .onnx per model, any hardware
    MPLBACKEND=Agg python -m gluefactory.scripts.run_inference \\
        --backend onnx --matcher lightglue \\
        --extractor_onnx models/superpoint.onnx \\
        --matcher_onnx   models/lightglue.onnx \\
        --execution_provider auto \\
        --input data/MAP2/images/rgb \\
        --output all --output_dir data/MAP2/visualizations/sp_lg_onnx
"""

import argparse
import json
import logging
import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from gluefactory.models.extractors.superpoint_open import (
    batched_nms,
    sample_descriptors,
    select_top_k_keypoints,
)
from gluefactory.slam.io import get_image_pairs, list_images, load_image
from gluefactory.slam.matcher import SLAMMatcher, run_match_directory
from gluefactory.visualization import dashboard, viz2d

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_inference")


# ─────────────────────────────────────────────────────────────────────────────
# Exported-model (.pt) backend
# ─────────────────────────────────────────────────────────────────────────────

class ExportedMatcher:
    """Checkpoint-free inference backend: loads .pt TorchScript graphs from
    gluefactory.scripts.trace_model and decodes SuperPoint's dense heatmap
    with the real training-time NMS/threshold/top-k/descriptor-sampling
    logic, rather than a crude np.where + top-k re-implementation.

    Exposes the same `match_pair`/`match_directory` interface as
    `gluefactory.slam.matcher.SLAMMatcher` so both backends can be driven
    identically from `main()` below.
    """

    def __init__(self, extractor_pt, matcher_pt=None, device=None, conf=None,
                 matcher="superglue"):
        matcher = matcher or "none"
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.has_matcher = matcher != "none"
        conf = conf or {}
        self.nms_radius = conf.get("nms_radius", 3)
        self.max_num_keypoints = conf.get("max_num_keypoints", 512)
        self.detection_threshold = conf.get("detection_threshold", 0.005)
        self.remove_borders = conf.get("remove_borders", 4)
        self.filter_threshold = conf.get("filter_threshold", 0.01)

        self._sp = torch.jit.load(str(extractor_pt), map_location=self.device).eval()
        self._matcher = None
        if self.has_matcher:
            if matcher_pt is None:
                raise ValueError("matcher_pt is required unless matcher='none'")
            self._matcher = torch.jit.load(str(matcher_pt), map_location=self.device).eval()
        logger.info(f"ExportedMatcher ({matcher}) loaded on {self.device}")

    def _extract(self, img_gray):
        """Decode one image: keypoints [N,2], scores [N], descriptors [1,256,N],
        and the [1,1,H,W] input tensor (needed by the matcher's image0/1 input).
        """
        t = torch.from_numpy(img_gray).float().to(self.device).unsqueeze(0).unsqueeze(0) / 255.0
        with torch.no_grad():
            out = self._sp({"image": t})
        scores = batched_nms(out["scores"], self.nms_radius)
        if self.remove_borders:
            pad = self.remove_borders
            scores[:, :pad] = -1
            scores[:, :, :pad] = -1
            scores[:, -pad:] = -1
            scores[:, :, -pad:] = -1
        scores = scores.squeeze(0)
        idxs = torch.where(scores > self.detection_threshold)
        kpts = torch.stack(idxs[-2:], dim=-1).flip(1).float()
        kscores = scores[idxs]
        kpts, kscores = select_top_k_keypoints(kpts, kscores, self.max_num_keypoints)
        with torch.no_grad():
            desc = sample_descriptors(kpts.unsqueeze(0), out["descriptors"], s=8)
        return kpts, kscores, desc, t

    def extract(self, img_gray):
        """Run the extractor alone on a single image (no matcher forward)."""
        kpts, kscores, _, _ = self._extract(img_gray)
        return {"keypoints": kpts.cpu().numpy(), "scores": kscores.cpu().numpy()}

    def match_pair(self, img0_gray, img1_gray):
        kpts0, sc0, desc0, t0 = self._extract(img0_gray)
        kpts1, sc1, desc1, t1 = self._extract(img1_gray)
        n0 = len(kpts0)

        if not self.has_matcher:
            matches0 = np.full(n0, -1, dtype=np.int64)
            mscores0 = np.zeros(n0, dtype=np.float32)
        else:
            with torch.no_grad():
                out = self._matcher({
                    "keypoints0": kpts0.unsqueeze(0),
                    "keypoints1": kpts1.unsqueeze(0),
                    "descriptors0": desc0,
                    "descriptors1": desc1,
                    "scores0": sc0.unsqueeze(0),
                    "scores1": sc1.unsqueeze(0),
                    "image0": t0,
                    "image1": t1,
                })
            matches0 = out["matches0"][0].cpu().numpy()
            mscores0 = out["matching_scores0"][0].cpu().numpy()

        return {
            "keypoints0": kpts0.cpu().numpy(),
            "keypoints1": kpts1.cpu().numpy(),
            "scores0": sc0.cpu().numpy(),
            "scores1": sc1.cpu().numpy(),
            "matches0": matches0,
            "mscores0": mscores0,
        }

    def match_directory(self, input_dir, max_pairs=None, resize=0):
        return run_match_directory(self, self.filter_threshold, input_dir, max_pairs, resize)


# ─────────────────────────────────────────────────────────────────────────────
# ONNX backend (onnxruntime): one .onnx per model, hardware-neutral.
# ─────────────────────────────────────────────────────────────────────────────

_EXECUTION_PROVIDERS = {
    "cpu": ("CPUExecutionProvider", {}),
    "cuda": ("CUDAExecutionProvider", {}),
    "openvino": ("OpenVINOExecutionProvider", {"device_type": "GPU"}),
}

_PROVIDER_INSTALL = {
    "CUDAExecutionProvider": "onnxruntime-gpu",
    "OpenVINOExecutionProvider": "onnxruntime-openvino",
    "CPUExecutionProvider": "onnxruntime",
}


def _resolve_providers(kind, available):
    """Return an onnxruntime provider chain for `kind` (auto|cpu|cuda|openvino).

    `available` is ort.get_available_providers(), which reflects the installed
    onnxruntime build (plain, -gpu, or -openvino). CUDA targets NVIDIA GPUs,
    OpenVINO (device_type=GPU) targets Intel iGPU/Arc on Linux.
    """
    if kind == "auto":
        if "CUDAExecutionProvider" in available:
            kind = "cuda"
        elif "OpenVINOExecutionProvider" in available:
            kind = "openvino"
        else:
            kind = "cpu"
    name, opts = _EXECUTION_PROVIDERS[kind]
    if name not in available:
        raise SystemExit(
            f"Execution provider {name!r} is not available in this onnxruntime build "
            f"(available: {available}). Install via: pip install "
            f"{_PROVIDER_INSTALL.get(name, 'onnxruntime')}"
        )
    if name == "CPUExecutionProvider":
        return [("CPUExecutionProvider", {})]
    return [(name, opts), ("CPUExecutionProvider", {})]


class OnnxMatcher:
    """onnxruntime backend: both networks are plain ONNX graphs, so the SAME
    model files run on any hardware by choosing an execution provider at load
    time (`--execution_provider`). SuperPoint's dense outputs are decoded with
    the same threshold/top-k/descriptor-sampling logic as `ExportedMatcher`,
    so results match the TorchScript path while staying device-agnostic.

    `match_directory` runs batch-wise on the GPU: every chunk of images goes
    through the extractor in ONE forward pass (its ONNX graph has a dynamic
    batch axis), while LightGlue stays per-pair (batch=1 by design, but only
    a few ms per pair on a GPU) -- turning ~5k pairs into a few minutes
    instead of ~30.
    """

    def __init__(self, extractor_onnx, matcher_onnx=None, provider="auto",
                 matcher="superglue", conf=None, batch_size=16, workers=8):
        import onnxruntime as ort

        matcher = matcher or "none"
        self.has_matcher = matcher != "none"
        conf = conf or {}
        self.max_num_keypoints = conf.get("max_num_keypoints", 512)
        self.detection_threshold = conf.get("detection_threshold", 0.005)
        self.remove_borders = conf.get("remove_borders", 4)
        self.filter_threshold = conf.get("filter_threshold", 0.01)
        self._batch_size = max(1, batch_size)
        self._decode_workers = max(1, workers)
        self._gray_cache = {}
        self._bgr_cache = {}

        providers = _resolve_providers(provider, ort.get_available_providers())
        logger.info(f"OnnxMatcher ({matcher}) execution providers: {providers}")

        # num_kpts varies per image; disable the memory arena / pattern so
        # onnxruntime doesn't try to reuse buffers across differently-shaped
        # runs (its default fails with "Shape mismatch attempting to re-use
        # buffer" when the dynamic axis actually changes between calls).
        def _session(model_path):
            so = ort.SessionOptions()
            so.enable_cpu_mem_arena = False
            so.enable_mem_pattern = False
            sess = ort.InferenceSession(
                str(model_path), sess_options=so, providers=providers
            )
            return sess

        self._sp = _session(extractor_onnx)
        self._sp_inputs = [i.name for i in self._sp.get_inputs()]

        self._matcher = None
        self._matcher_inputs = []
        if self.has_matcher:
            if matcher_onnx is None:
                raise ValueError("matcher_onnx is required unless matcher='none'")
            self._matcher = _session(matcher_onnx)
            self._matcher_inputs = [i.name for i in self._matcher.get_inputs()]

    def _decode_scores(self, scores, descriptors):
        """Decode ONE SuperPoint output plane into keypoints/scores/descriptors.

        `scores`/`descriptors` from the ONNX graph are already NMS-filtered,
        border-zeroed and L2-normalized (baked at export time); here we apply
        detection threshold, top-k and descriptor sampling.
        """
        scores = np.asarray(scores)  # [H,W]
        if self.remove_borders:
            p = self.remove_borders
            scores[:p] = -1.0
            scores[-p:] = -1.0
            scores[:, :p] = -1.0
            scores[:, -p:] = -1.0

        idxs = np.where(scores > self.detection_threshold)
        if len(idxs[0]) == 0:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((1, 256, 0), dtype=np.float32),
            )
        kpts = np.stack(idxs[-2:], axis=-1)[:, ::-1].astype(np.float32)  # [N,2] (x,y)
        kscores = scores[idxs].astype(np.float32)

        if len(kpts) > self.max_num_keypoints:
            order = np.argsort(-kscores, kind="stable")[: self.max_num_keypoints]
            kpts, kscores = kpts[order], kscores[order]

        with torch.no_grad():
            desc = sample_descriptors(
                torch.from_numpy(kpts[None]),
                torch.from_numpy(np.asarray(descriptors)),
                s=8,
            )  # [1,256,N]
        return kpts, kscores, desc

    def _extract(self, img_gray):
        """Decode one image: keypoints [N,2], scores [N], descriptors [1,256,N]."""
        t = (img_gray.astype(np.float32) / 255.0)[None, None]  # [1,1,H,W]
        scores, descriptors = self._sp.run(None, {self._sp_inputs[0]: t})
        kpts, kscores, desc = self._decode_scores(scores[0], descriptors)
        return kpts, kscores, desc, t

    def _extract_batch(self, grays):
        """Batch-decode a list of grayscale images with ONE extractor run.

        Returns a dict indexed by the input array's id() (shared image objects
        are decoded once, matching match_directory's path cache).
        """
        by_shape = {}
        for g in grays:
            by_shape.setdefault(g.shape, []).append(g)
        out = {}
        for shape, group in by_shape.items():
            t = np.asarray(group, dtype=np.float32)[:, None] / 255.0  # [B,1,H,W]
            scores, descriptors = self._sp.run(None, {self._sp_inputs[0]: t})
            for gi, g in enumerate(group):
                out[id(g)] = self._decode_scores(
                    scores[gi], descriptors[gi : gi + 1]
                )
        return out

    def extract(self, img_gray):
        """Run the extractor alone on a single image (no matcher forward)."""
        kpts, kscores, _, _ = self._extract(img_gray)
        return {"keypoints": kpts, "scores": kscores}

    def _match_views(self, kpts0, sc0, desc0, kpts1, sc1, desc1, w, h):
        """Single-pair matcher forward (no padding)."""
        size = np.array([[w, h]], dtype=np.float32)
        feed = {
            "keypoints0": kpts0[None].astype(np.float32),
            "keypoints1": kpts1[None].astype(np.float32),
            "descriptors0": desc0.numpy().astype(np.float32),
            "descriptors1": desc1.numpy().astype(np.float32),
            "size0": size,
            "size1": size,
        }
        if "scores0" in self._matcher_inputs:  # SuperGlue graphs
            feed["scores0"] = sc0[None].astype(np.float32)
            feed["scores1"] = sc1[None].astype(np.float32)
        out = self._matcher.run(None, feed)
        return np.asarray(out[0])[0], np.asarray(out[2])[0]

    def _match_serial(self, views):
        """Match a list of pairs one at a time.

        LightGlue is batch=1 by design (its positional encoding broadcasts
        the batch at axis 2, `LearnableFourierPositionalEncoding.unsqueeze(-3)`,
        so b>1 clashes with the head axis; the ONNX graph inherits that), so
        pairs go through the matcher sequentially. On a GPU each pair is only
        a few milliseconds -- the big batch win is extraction (below).
        """
        results = []
        for v in views:
            mo, ms = self._match_views(
                v["kpts0"], v["sc0"], v["desc0"],
                v["kpts1"], v["sc1"], v["desc1"], v["w"], v["h"],
            )
            results.append((mo, ms))
        return results

    def match_pair(self, img0_gray, img1_gray):
        kpts0, sc0, desc0, _ = self._extract(img0_gray)
        kpts1, sc1, desc1, _ = self._extract(img1_gray)
        n0 = len(kpts0)

        if not self.has_matcher:
            matches0 = np.full(n0, -1, dtype=np.int64)
            mscores0 = np.zeros(n0, dtype=np.float32)
        elif n0 == 0 or len(kpts1) == 0:
            matches0 = np.full(n0, -1, dtype=np.int64)
            mscores0 = np.zeros(n0, dtype=np.float32)
        else:
            h, w = img0_gray.shape
            matches0, mscores0 = self._match_views(
                kpts0, sc0, desc0, kpts1, sc1, desc1, w, h
            )

        return {
            "keypoints0": kpts0,
            "keypoints1": kpts1,
            "scores0": sc0,
            "scores1": sc1,
            "matches0": matches0,
            "mscores0": mscores0,
        }

    def _get_image(self, path, resize):
        """Decode (bgr, gray) for a path once; later pairs reuse the cache."""
        p = str(path)
        if p not in self._gray_cache:
            bgr, gray = load_image(path, resize)
            self._bgr_cache[p] = bgr
            self._gray_cache[p] = gray
        return self._bgr_cache[p], self._gray_cache[p]

    def match_directory(self, input_dir, max_pairs=None, resize=0):
        """Match all consecutive pairs batch-wise on the accelerator.

        For each chunk of `batch_size` pairs: the chunk's unique images are
        decoded in parallel worker threads, all of them go through the
        extractor in ONE forward pass, and all pairs through the matcher in
        ONE padded mini-batch forward. Produces the same record schema as
        `slam.matcher.run_match_directory`.
        """
        pairs = get_image_pairs(input_dir)
        if not pairs:
            logger.error(f"No image pairs found in: {input_dir}")
            return []
        if max_pairs:
            pairs = pairs[:max_pairs]
        logger.info(f"OnnxMatcher batch run: {len(pairs)} pairs, "
                    f"batch={self._batch_size}, decode_workers={self._decode_workers}")

        results = []
        bs = self._batch_size
        n_pairs = len(pairs)
        with ThreadPoolExecutor(max_workers=self._decode_workers) as pool:
            for start in tqdm(range(0, n_pairs, bs), desc="Batching"):
                chunk = pairs[start:start + bs]

                # 1) decode this chunk's unique, not-yet-cached images in parallel
                uniq = []
                for p0, p1 in chunk:
                    for p in (p0, p1):
                        if str(p) not in self._gray_cache:
                            uniq.append(p)
                futs = [pool.submit(self._get_image, p, resize) for p in uniq]
                for f in futs:
                    f.result()

                # 2) batch-extract every image in the chunk (shared objects once)
                grays = [self._gray_cache[str(p)] for p0, p1 in chunk for p in (p0, p1)]
                feats = self._extract_batch(grays)

                # 3) match the pairs (LightGlue is per-pair by design)
                views = []
                for p0, p1 in chunk:
                    g0 = self._gray_cache[str(p0)]
                    g1 = self._gray_cache[str(p1)]
                    k0, sc0, d0 = feats[id(g0)]
                    k1, sc1, d1 = feats[id(g1)]
                    h, w = g0.shape
                    views.append(
                        {"kpts0": k0, "sc0": sc0, "desc0": d0,
                         "kpts1": k1, "sc1": sc1, "desc1": d1, "w": w, "h": h}
                    )
                if self.has_matcher:
                    pair_outs = self._match_serial(views)
                else:
                    pair_outs = None

                # 4) assemble records (identical schema to run_match_directory)
                for i, (p0, p1) in enumerate(chunk):
                    v = views[i]
                    if pair_outs is None:
                        m0 = np.full(len(v["kpts0"]), -1, dtype=np.int64)
                        ms0 = np.zeros(len(v["kpts0"]), dtype=np.float32)
                    else:
                        m0, ms0 = pair_outs[i]
                    valid = (m0 != -1) & (ms0 >= self.filter_threshold)
                    n_match = int(valid.sum())
                    n0, n1 = len(v["kpts0"]), len(v["kpts1"])
                    avg_s = float(ms0[valid].mean()) if n_match > 0 else 0.0
                    results.append({
                        "idx": start + i,
                        "path0": p0,
                        "path1": p1,
                        "name0": p0.name,
                        "name1": p1.name,
                        "img0_bgr": self._bgr_cache[str(p0)],
                        "img1_bgr": self._bgr_cache[str(p1)],
                        "keypoints0": v["kpts0"],
                        "keypoints1": v["kpts1"],
                        "scores0": v["sc0"],
                        "scores1": v["sc1"],
                        "matches0": m0,
                        "mscores0": ms0,
                        "metrics": {
                            "total_kpts0": n0,
                            "total_kpts1": n1,
                            "num_matches": n_match,
                            "match_ratio": n_match / max(1, min(n0, n1)),
                            "avg_mscore": avg_s,
                        },
                    })
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Extractor-only (--matcher none) driver: one pass per image, not per pair.
# ─────────────────────────────────────────────────────────────────────────────

def _run_extractor_only(backend, input_path, output_dir, outputs, resize, max_pairs):
    image_paths = list_images(input_path)
    if not image_paths:
        logger.error(f"No images found in: {input_path}")
        return
    if max_pairs:
        image_paths = image_paths[:max_pairs]

    png_dir = output_dir / "images"
    if "png" in outputs or "html" in outputs:
        png_dir.mkdir(parents=True, exist_ok=True)

    records = []
    png_paths = []
    for idx, p in enumerate(tqdm(image_paths, desc="Extracting")):
        img_bgr, img_gray = load_image(p, resize)
        out = backend.extract(img_gray)
        kpts, scores = out["keypoints"], out["scores"]
        logger.info(f"[{idx}] {p.name}: {len(kpts)} keypoints")

        records.append({
            "idx": idx, "name": p.name,
            "keypoints": kpts.tolist(), "scores": scores.tolist(),
            "num_keypoints": int(len(kpts)),
        })

        if "png" in outputs or "html" in outputs:
            png_path = png_dir / f"{p.stem}_keypoints.png"
            viz2d.plot_keypoints_overlay(
                cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), kpts, scores, png_path,
                title=f"{p.name} ({len(kpts)} kpts)", colorbar=True,
            )
            png_paths.append(png_path)

    if "data" in outputs:
        data_path = output_dir / "keypoints_data.json"
        data_path.write_text(json.dumps(records, indent=2))
        logger.info(f"Saved {len(records)} record(s) → {data_path}")

    if "html" in outputs:
        dashboard.render_image_grid(png_dir, png_paths, title="Keypoint Visualizer")

    print(f"\nTotal processed images: {len(records)}")
    print(f"Results: {output_dir.resolve()}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Pair (--matcher superglue|lightglue) driver
# ─────────────────────────────────────────────────────────────────────────────

def _save_worker(task):
    """Write one pair's view images and/or match plot, off the main thread."""
    (bgr0, bgr1, kpts0, kpts1, matches0, mscores0,
     view0, view1, plot, title) = task
    if view0 is not None:
        cv2.imwrite(str(view0), bgr0)
        cv2.imwrite(str(view1), bgr1)
    if plot is not None:
        img0_rgb = cv2.cvtColor(bgr0, cv2.COLOR_BGR2RGB)
        img1_rgb = cv2.cvtColor(bgr1, cv2.COLOR_BGR2RGB)
        viz2d.plot_pair_matches(
            img0_rgb, img1_rgb, kpts0, kpts1, matches0, mscores0,
            str(plot), title=title,
        )
    return None


def _run_pairs(
    backend, input_path, output_dir, outputs, resize, max_pairs, matcher_name,
    save_workers=1, log_every=0, downsample=1,
):
    results = backend.match_directory(input_path, max_pairs=max_pairs, resize=resize)
    if not results:
        logger.error("No pairs were matched.")
        return

    images_dir = output_dir / "images"
    plots_dir = output_dir / "plots"
    # "data" mode still needs the raw view images on disk: matches_data.js
    # references them by URL for any external HTML viewer to load.
    need_images = "html" in outputs or "data" in outputs
    need_plots = "png" in outputs
    if need_images:
        images_dir.mkdir(parents=True, exist_ok=True)
    if need_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    # matplotlib + PNG encoding for ~5k pairs is the bulk of a batch run's
    # wall time, so push imwrites/plots onto a process pool when asked.
    use_pool = save_workers > 1 and (need_plots or need_images)
    pool = multiprocessing.Pool(save_workers) if use_pool else None
    sema = threading.BoundedSemaphore(max(2 * save_workers, 8)) if use_pool else None

    def submit(task):
        if pool is None:
            _save_worker(task)
        else:
            sema.acquire()
            pool.apply_async(_save_worker, (task,),
                             callback=lambda _: sema.release())

    records = []
    n_queued = 0
    for r in tqdm(results, desc="Saving"):
        idx = r["idx"]
        record = {
            "idx": idx,
            "name0": r["name0"],
            "name1": r["name1"],
            "keypoints0": r["keypoints0"].tolist(),
            "keypoints1": r["keypoints1"].tolist(),
            "scores0": r["scores0"].tolist(),
            "scores1": r["scores1"].tolist(),
            "matches": [
                [int(i0), int(i1), float(r["mscores0"][i0])]
                for i0, i1 in enumerate(r["matches0"]) if i1 != -1
            ],
            "metrics": r["metrics"],
            "image_width": int(r["img0_bgr"].shape[1]),
            "image_height": int(r["img0_bgr"].shape[0]),
        }

        view0 = view1 = plot = None
        if need_images:
            n0, n1 = f"pair_{idx}_view0.png", f"pair_{idx}_view1.png"
            view0, view1 = images_dir / n0, images_dir / n1
            record["image0_url"] = f"images/{n0}"
            record["image1_url"] = f"images/{n1}"
        if need_plots:
            plot_name = f"pair_{idx}_matches.png"
            plot = plots_dir / plot_name
            record["plot_url"] = f"plots/{plot_name}"

        if view0 is not None or plot is not None:
            submit((
                r["img0_bgr"], r["img1_bgr"],
                np.ascontiguousarray(r["keypoints0"]),
                np.ascontiguousarray(r["keypoints1"]),
                np.ascontiguousarray(r["matches0"]),
                np.ascontiguousarray(r["mscores0"]),
                view0, view1, plot,
                f"{r['name0']} ↔ {r['name1']}",
            ))
            n_queued += 1

        if log_every and (len(records) % log_every == 0):
            m = record["metrics"]
            logger.info(
                f"Pair {idx}: {r['name0']} ↔ {r['name1']}  "
                f"kpts {m['total_kpts0']}/{m['total_kpts1']}  "
                f"matches {m['num_matches']} ({m['match_ratio']:.1%})  "
                f"avg {m['avg_mscore']:.2f}"
            )
        records.append(record)

    if pool is not None:
        pool.close()
        pool.join()

    dash_records = records[::downsample] if downsample > 1 else records
    if "html" in outputs:
        # render_dashboard writes matches_data.js itself; avoid writing it twice.
        title = f"SuperPoint + {matcher_name.capitalize()} Visualizer"
        html_path = dashboard.render_dashboard(output_dir, dash_records, title=title)
        print(f"Dashboard: file://{html_path.resolve()}")
    elif "data" in outputs:
        dashboard.write_matches_data(output_dir, dash_records)

    if downsample > 1:
        print(f"Dashboard shows {len(dash_records)}/{len(records)} pairs "
              f"(every {downsample}th)")
    print(f"\nTotal processed pairs: {len(records)}")
    print(f"Results: {output_dir.resolve()}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

_OUTPUT_ALIASES = {"all": ("html", "data", "png")}


def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backend", choices=["checkpoint", "exported", "onnx"], default="checkpoint"
    )
    parser.add_argument("--matcher", choices=["none", "superglue", "lightglue"],
                         default="superglue")

    parser.add_argument("--extractor_ckpt", type=str, default=None,
                         help="SuperPoint training checkpoint (.tar); required for "
                              "--backend checkpoint")
    parser.add_argument("--matcher_ckpt", type=str, default=None,
                         help="Matcher training checkpoint (.tar); required for "
                              "--backend checkpoint unless --matcher none")
    parser.add_argument("--extractor_pt", type=str, default=None,
                         help="Exported SuperPoint model (.pt); required for "
                              "--backend exported")
    parser.add_argument("--matcher_pt", type=str, default=None,
                         help="Exported matcher model (.pt); required for "
                              "--backend exported unless --matcher none")
    parser.add_argument("--extractor_onnx", type=str, default=None,
                         help="SuperPoint model (.onnx); required for --backend onnx")
    parser.add_argument("--matcher_onnx", type=str, default=None,
                         help="Matcher model (.onnx); required for --backend onnx "
                              "unless --matcher none")
    parser.add_argument(
        "--execution_provider", choices=["auto", "cpu", "cuda", "openvino"],
        default="auto",
        help="onnxruntime execution provider (--backend onnx): auto picks "
             "CUDA > OpenVINO > CPU from what is installed.",
    )

    parser.add_argument("--input", type=str, required=True,
                         help="Image directory, single image, or 'img0.png,img1.png'")
    parser.add_argument("--output", nargs="+",
                         choices=["html", "data", "png", "all"], default=["all"],
                         help="Combinable: html, data, png (or 'all'). For 5k+ "
                              "batches use 'html data' for a fast dashboard "
                              "(no per-pair plots); add png + --save_workers "
                              "to render plots too.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resize", type=int, default=640)
    parser.add_argument("--max_pairs", type=int, default=0, help="Cap on pairs/images (0 = all)")

    parser.add_argument("--batch_size", type=int, default=16,
                         help="Pairs per chunk: images in a chunk go through "
                              "the extractor in one batched forward "
                              "(ONNX backend only)")
    parser.add_argument("--workers", type=int, default=8,
                         help="Parallel image-decoding threads (ONNX backend only)")
    parser.add_argument("--save_workers", type=int, default=1,
                         help="Process pool for saving view images + match plots "
                              "(0 disables pool; e.g. 8 for 5k+ batches)")
    parser.add_argument("--log_every", type=int, default=0,
                         help="Log one pair metric line every N pairs (0 = every pair)")
    parser.add_argument("--downsample_dashboard", type=int, default=1,
                         help="Keep only every Nth pair in the dashboard/data file "
                              "(1 = all; e.g. 3 for ~5k pairs)")

    parser.add_argument("--nms_radius", type=int, default=3)
    parser.add_argument("--max_num_keypoints", type=int, default=512)
    parser.add_argument("--detection_threshold", type=float, default=0.005)
    parser.add_argument("--filter_threshold", type=float, default=0.01)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = _parse_args()
    outputs = []
    for o in args.output:
        outputs.extend(_OUTPUT_ALIASES.get(o, (o,)))
    outputs = tuple(dict.fromkeys(outputs))  # dedupe, keep order

    conf = {
        "nms_radius": args.nms_radius,
        "max_num_keypoints": args.max_num_keypoints,
        "detection_threshold": args.detection_threshold,
        "filter_threshold": args.filter_threshold,
    }

    if args.backend == "checkpoint":
        if not args.extractor_ckpt:
            raise SystemExit("--extractor_ckpt is required for --backend checkpoint")
        if args.matcher != "none" and not args.matcher_ckpt:
            raise SystemExit("--matcher_ckpt is required unless --matcher none")
        backend = SLAMMatcher(
            args.matcher_ckpt, args.extractor_ckpt, device=args.device, conf=conf,
            matcher=args.matcher,
        )
    elif args.backend == "onnx":
        if not args.extractor_onnx:
            raise SystemExit("--extractor_onnx is required for --backend onnx")
        if args.matcher != "none" and not args.matcher_onnx:
            raise SystemExit("--matcher_onnx is required unless --matcher none")
        backend = OnnxMatcher(
            args.extractor_onnx, args.matcher_onnx,
            provider=args.execution_provider, matcher=args.matcher, conf=conf,
            batch_size=args.batch_size, workers=args.workers,
        )
    else:
        if not args.extractor_pt:
            raise SystemExit("--extractor_pt is required for --backend exported")
        if args.matcher != "none" and not args.matcher_pt:
            raise SystemExit("--matcher_pt is required unless --matcher none")
        backend = ExportedMatcher(
            args.extractor_pt, args.matcher_pt, device=args.device, conf=conf,
            matcher=args.matcher,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_pairs = args.max_pairs if args.max_pairs > 0 else None

    if args.matcher == "none":
        _run_extractor_only(
            backend, args.input, output_dir, outputs, args.resize, max_pairs
        )
    else:
        _run_pairs(
            backend, args.input, output_dir, outputs, args.resize, max_pairs,
            args.matcher, save_workers=args.save_workers,
            log_every=args.log_every, downsample=args.downsample_dashboard,
        )


if __name__ == "__main__":
    main()
