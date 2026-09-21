#!/usr/bin/env python3
"""
Numerically verify an exported LightGlue/SuperGlue .onnx against the reference
PyTorch model loaded from the original .tar checkpoint.

Feeds identical random inputs to both the ONNX session (onnxruntime) and the
traced-matcher reference built by trace_model.py, then compares:
  * matches0 / matches1      -> exact match rate (integer indices, -1 = unmatched)
  * matching_scores0         -> max abs difference and mean abs difference

Usage:
  python -m gluefactory.scripts.verify_onnx \
      --onnx lightglue.onnx \
      --ckpt outputs/training/lightglue_slam_run/checkpoint_best.tar \
      --model lightglue --num_kpts 512 --image_h 480 --image_w 640
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from gluefactory.scripts.trace_model import (
    _build_superglue,
    _build_lightglue,
    SuperGlueONNX,
    LightGlueONNX,
)


def _make_inputs(model, num_kpts: int, image_h: int, image_w: int, seed: int = 0):
    torch.manual_seed(seed)
    B, N = 1, num_kpts

    if model == "superglue":
        D = 256
        descriptors = torch.rand(B, D, N).float()
        scores0 = torch.rand(B, N).float()
        scores1 = torch.rand(B, N).float()
    else:
        D = 256
        descriptors = torch.rand(B, D, N).float()

    kpts0 = torch.rand(B, N, 2).float() * torch.tensor([image_w, image_h]).float()
    kpts1 = torch.rand(B, N, 2).float() * torch.tensor([image_w, image_h]).float()
    size = torch.tensor([[float(image_w), float(image_h)]]).expand(B, -1)

    if model == "superglue":
        inputs = {
            "keypoints0": kpts0,
            "keypoints1": kpts1,
            "descriptors0": descriptors,
            "descriptors1": descriptors,
            "scores0": scores0,
            "scores1": scores1,
            "size0": size,
            "size1": size,
        }
    else:
        inputs = {
            "keypoints0": kpts0,
            "keypoints1": kpts1,
            "descriptors0": descriptors,
            "descriptors1": descriptors,
            "size0": size,
            "size1": size,
        }
    return inputs


def _build_reference(model, ckpt_path: Path, num_kpts: int, image_h: int, image_w: int):
    if model == "superglue":
        sg = _build_superglue(ckpt_path)
        wrapper = SuperGlueONNX(sg, 50).eval()
        ref_inputs = _make_inputs("superglue", num_kpts, image_h, image_w)
    else:
        lg = _build_lightglue(ckpt_path)
        wrapper = LightGlueONNX(lg, lg.conf.n_layers).eval()
        ref_inputs = _make_inputs("lightglue", num_kpts, image_h, image_w)
    return wrapper, ref_inputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", required=True, type=Path, help="Paths to the exported .onnx")
    p.add_argument("--ckpt", required=True, type=Path, help=".tar checkpoint of the reference model")
    p.add_argument("--model", choices=["superglue", "lightglue"], required=True)
    p.add_argument("--num_kpts", type=int, default=512)
    p.add_argument("--image_h", type=int, default=480)
    p.add_argument("--image_w", type=int, default=640)
    p.add_argument("--tol", type=float, default=1e-3,
                   help="max allowed abs diff on matching_scores0 (default: %(default)s)")
    args = p.parse_args()

    import onnxruntime as ort

    if not args.onnx.exists():
        print(f"Error: onnx model not found: {args.onnx}", file=sys.stderr)
        sys.exit(1)
    if not args.ckpt.exists():
        print(f"Error: checkpoint not found: {args.ckpt}", file=sys.stderr)
        sys.exit(1)

    print(f"[verify] onnx    = {args.onnx}")
    print(f"[verify] ckpt    = {args.ckpt}")
    print(f"[verify] model   = {args.model}  kpts={args.num_kpts} "
          f"image={args.image_w}x{args.image_h}")

    wrapper, ref_inputs = _build_reference(
        args.model, args.ckpt, args.num_kpts, args.image_h, args.image_w
    )

    ref_args = tuple(ref_inputs[k] for k in (
        ("keypoints0", "keypoints1", "descriptors0", "descriptors1",
         "scores0", "scores1", "size0", "size1")
        if args.model == "superglue" else
        ("keypoints0", "keypoints1", "descriptors0", "descriptors1",
         "size0", "size1")
    ))

    with torch.no_grad():
        m0_t, m1_t, msc0_t, msc1_t = wrapper(*ref_args)

    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    onnx_feed = {k: v.numpy() for k, v in ref_inputs.items()}
    outs = sess.run(None, onnx_feed)
    names = [o.name for o in sess.get_outputs()]
    out_map = dict(zip(names, outs))
    m0_o = torch.from_numpy(out_map["matches0"])
    m1_o = torch.from_numpy(out_map["matches1"])
    msc0_o = torch.from_numpy(out_map["matching_scores0"])

    match_rate0 = (m0_t == m0_o).float().mean().item() * 100.0
    match_rate1 = (m1_t == m1_o).float().mean().item() * 100.0
    max_diff = (msc0_t - msc0_o).abs().max().item()
    mean_diff = (msc0_t - msc0_o).abs().mean().item()

    print(f"[verify] matches0 exact-match rate : {match_rate0:.3f}% ({args.num_kpts} kpts)")
    print(f"[verify] matches1 exact-match rate : {match_rate1:.3f}%")
    print(f"[verify] matching_scores0 max diff : {max_diff:.6e}")
    print(f"[verify] matching_scores0 mean diff: {mean_diff:.6e}")

    if max_diff <= args.tol and match_rate0 >= 99.99 and match_rate1 >= 99.99:
        print("[verify] PASS")
    else:
        print(f"[verify] FAIL (tol={args.tol})")
        sys.exit(1)


if __name__ == "__main__":
    main()