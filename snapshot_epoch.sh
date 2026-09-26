#!/usr/bin/env bash
# Wait for a specific training epoch's checkpoint to appear, run SuperPoint-only
# inference from it, and record the result as a git commit on a snapshots branch
# -- while training continues in the foreground.
#
# Motivation: a 100-epoch MAP1 run is long, and the interesting question is
# "does the detector still look sane at epoch 10?". Waiting for the whole run
# to finish gives no early signal, and killing training to poke at an
# intermediate checkpoint is wasteful. This watches for the epoch checkpoint
# instead and snapshots it.
#
# Usage:
#   bash snapshot_epoch.sh <experiment> <epoch> [data_dir] [threshold] \
#                          [max_pairs] [branch] [png_count]
#
#   experiment   experiment name passed to gluefactory.train
#   epoch        epoch number to wait for (0-indexed, as train.py logs it)
#   data_dir     dataset dir under data/            (default MAP1)
#   threshold    detection_threshold for inference   (default 0.005)
#   max_pairs    pairs to render                     (default 50)
#   branch       branch to record the commit on      (default snapshots/<experiment>)
#   png_count    how many rendered PNGs to commit    (default 10)
#
# run_full_pipeline.sh launches this automatically for every epoch listed in
# SNAPSHOT_EPOCHS. It can also be run by hand against an existing experiment.
#
# How the git recording works: the commit is built with plumbing against a
# temporary index, then pushed onto refs/heads/<branch> with update-ref. That
# writes a commit without ever touching the working tree, the real index, or
# HEAD -- so a snapshot can be taken while training is writing checkpoints into
# the same directory, and without leaving uncommitted changes behind.

set -euo pipefail

EXP="${1:?usage: snapshot_epoch.sh <experiment> <epoch> [data_dir] [threshold] [max_pairs] [branch] [png_count]}"
EP="${2:?missing epoch}"
DATA_DIR="${3:-MAP1}"
THRESHOLD="${4:-0.005}"
MAX_PAIRS="${5:-50}"
BRANCH="${6:-snapshots/${EXP}}"
PNG_COUNT="${7:-10}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$REPO_ROOT"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

EXP_DIR="outputs/training/${EXP}"
LOG="${EXP_DIR}/log.txt"
CKPT_DIR="outputs/snapshots/${EXP}_ep${EP}"
VIZ_DIR="${CKPT_DIR}/viz"
SNAP_DIR="snapshots/${EXP}_ep${EP}"
DONE_MARKER="${CKPT_DIR}/.snapshot_done"

mkdir -p "$CKPT_DIR" "$SNAP_DIR"

echo "[$(date)] snapshot_epoch: waiting for epoch ${EP} of ${EXP}"
echo "[$(date)]   checkpoint dir: ${EXP_DIR}"
echo "[$(date)]   target branch:  ${BRANCH}"

# --- wait for the epoch checkpoint ------------------------------------------
# train.py's end-of-epoch save (train.py:655) is unconditional, so a
# checkpoint_{epoch}_{iter}.tar is written for every epoch, not just the ones
# that happen to hit train.save_every_iter. Prefer the exact epoch match and
# fall back to the running best if training was interrupted before reaching it.
# The loop also exits if training dies, so a typo'd epoch can't hang forever.
CKPT=""
while true; do
    if [ ! -d "$EXP_DIR" ]; then
        echo "[$(date)] snapshot_epoch: ${EXP_DIR} does not exist yet, waiting"
    else
        CKPT="$(ls -1 "$EXP_DIR"/checkpoint_"${EP}"_*.tar 2>/dev/null | head -1 || true)"
        if [ -z "$CKPT" ] && [ -f "${EXP_DIR}/checkpoint_best.tar" ] && [ -f "$LOG" ]; then
            LAST_EP="$(grep -oE 'Starting epoch [0-9]+' "$LOG" | tail -1 | grep -oE '[0-9]+' || echo 0)"
            if [ "$LAST_EP" -gt "$EP" ]; then
                echo "[$(date)] snapshot_epoch: training already passed epoch ${EP}" \
                     "(last started ${LAST_EP}), using checkpoint_best.tar instead"
                CKPT="${EXP_DIR}/checkpoint_best.tar"
                break
            fi
        fi
        if [ -n "$CKPT" ]; then
            break
        fi
    fi
    if [ -f "outputs/training/${EXP}.DONE" ] || [ -f "${EXP_DIR}/log.txt" -a -f "${EXP_DIR}/.interrupted" ]; then
        if [ -f "${EXP_DIR}/checkpoint_best.tar" ]; then
            echo "[$(date)] snapshot_epoch: training ended before epoch ${EP}, using checkpoint_best.tar"
            CKPT="${EXP_DIR}/checkpoint_best.tar"
            break
        fi
        echo "[$(date)] snapshot_epoch: training ended and no checkpoint exists, giving up"
        exit 1
    fi
    sleep 30
done

echo "[$(date)] snapshot_epoch: using checkpoint ${CKPT}"
# Copy it out before inferring: keep_last_checkpoints prunes old checkpoints, so
# the original could be deleted while inference is still reading it.
cp "$CKPT" "${CKPT_DIR}/$(basename "$CKPT")"
CKPT_SNAP="${CKPT_DIR}/$(basename "$CKPT")"

# --- inference --------------------------------------------------------------
rm -rf "$VIZ_DIR"
echo "[$(date)] snapshot_epoch: rendering ${MAX_PAIRS} pairs at threshold ${THRESHOLD}"
MPLBACKEND=Agg python3 -m gluefactory.scripts.run_inference \
    --backend checkpoint --matcher none \
    --extractor_ckpt "$CKPT_SNAP" \
    --input "data/${DATA_DIR}/images/rgb" \
    --output all \
    --output_dir "$VIZ_DIR" \
    --resize 0 \
    --nms_radius 4 \
    --detection_threshold "$THRESHOLD" \
    --max_pairs "$MAX_PAIRS" \
    --log_every 100

# --- stats ------------------------------------------------------------------
# kpts/frame is the signal that actually separates a healthy detector from a
# collapsed one. loss/detector is useless here: with a frozen detector it is a
# large constant (7669 on Ouster, 12660 on MAP1) that never trains, so it says
# nothing about health either way.
STATS="${SNAP_DIR}/stats.txt"
python3 - "$VIZ_DIR/keypoints_data.json" "$STATS" "$EXP" "$EP" "$THRESHOLD" <<'PY'
import json, sys, statistics as st
import numpy as np

src, dst, exp, ep, thresh = sys.argv[1:6]
recs = json.load(open(src))
counts, tops, ux, uy, spacing = [], [], [], [], []
for r in recs:
    k = np.asarray(r["keypoints"], dtype=float)
    counts.append(len(k))
    if not len(k):
        continue
    ux.append(np.round(k[:, 0])); uy.append(np.round(k[:, 1]))
    if r.get("scores"):
        tops.append(max(r["scores"]))
    d = np.linalg.norm(k[:, None, :] - k[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    spacing.append(d.min(axis=1))

total = int(sum(counts))
per_frame = st.mean(counts)
ux_n = len(np.unique(np.concatenate(ux))) if ux else 0
uy_n = len(np.unique(np.concatenate(uy))) if uy else 0
top_p05 = float(np.percentile(tops, 5)) if tops else float("nan")
sp = float(np.percentile(np.concatenate(spacing), 50)) if spacing else float("nan")

# Collapse shows up as almost no detections, or as a saturated heatmap. Note
# this deliberately does NOT use lattice geometry: x-repeat% and nn_p50 sit at
# the same values for healthy and collapsed runs alike, so keying off them
# flags stock SuperPoint as broken.
if per_frame < 5:
    verdict = "DEGENERATE (fewer than 5 keypoints/frame)"
elif top_p05 > 0.95:
    verdict = "SUSPECT (top scores saturated near 1.0 -- heatmap is flat)"
elif per_frame < 50:
    verdict = "LOW (under 50 keypoints/frame -- check detection_threshold)"
else:
    verdict = "OK (healthy-looking spread of keypoints)"

with open(dst, "w") as f:
    f.write(f"experiment        {exp}\n")
    f.write(f"epoch             {ep}\n")
    f.write(f"detection_threshold {thresh}\n")
    f.write(f"images            {len(counts)}\n")
    f.write(f"keypoints total   {total}\n")
    f.write(f"keypoints/frame   {per_frame:.1f}\n")
    f.write(f"unique_x          {ux_n}\n")
    f.write(f"unique_y          {uy_n}\n")
    f.write(f"nn_spacing_p50    {sp:.2f}\n")
    f.write(f"top_score_p05     {top_p05:.3f}\n")
    f.write(f"verdict           {verdict}\n")
print(open(dst).read())
PY

# --- stage a small, committable subset --------------------------------------
# The .tar checkpoint stays out of git (large binary); only the gallery and a
# few sample frames are recorded.
cp "$VIZ_DIR/keypoints_data.json" "$SNAP_DIR/" 2>/dev/null || true
cp "$VIZ_DIR/images/index.html" "$SNAP_DIR/index.html" 2>/dev/null || true
mkdir -p "$SNAP_DIR/images"
find "$VIZ_DIR/images" -name '*_keypoints.png' | sort | head -"$PNG_COUNT" \
    | while read -r p; do cp "$p" "$SNAP_DIR/images/"; done

# --- record as a commit without touching the working tree -------------------
# A temporary index plus commit-tree/update-ref: HEAD, the real index and every
# tracked file stay exactly as they are, so this is safe to run while training
# writes into outputs/.
if [ ! -e "$DONE_MARKER" ]; then
    IDX="$(mktemp)"
    trap 'rm -f "$IDX"' EXIT
    # Base the new commit on the branch tip when it already exists, not on HEAD.
    # Reading HEAD here would silently drop every earlier snapshot: the branch
    # would be rebuilt from main each time, so a second snapshot (epoch 50)
    # would erase the first (epoch 10) from refs/heads/<branch>.
    if git rev-parse --verify --quiet "refs/heads/${BRANCH}" >/dev/null; then
        BASE="$(git rev-parse "refs/heads/${BRANCH}")"
    else
        BASE="$(git rev-parse HEAD)"
    fi
    GIT_INDEX_FILE="$IDX" git read-tree "$BASE"
    GIT_INDEX_FILE="$IDX" git add -f "$SNAP_DIR"
    TREE="$(GIT_INDEX_FILE="$IDX" git write-tree)"
    SUMMARY="$(grep -E 'keypoints/frame|top_score_p05|verdict' "$STATS" | tr '\n' ' ')"
    COMMIT="$(git commit-tree "$TREE" -p "$BASE" \
        -m "snapshot: ${EXP} epoch ${EP} inference (${SUMMARY})")"
    git update-ref "refs/heads/${BRANCH}" "$COMMIT"
    touch "$DONE_MARKER"
    echo "[$(date)] snapshot_epoch: committed ${COMMIT:0:8} on ${BRANCH} (parent ${BASE:0:8})"
    echo "[$(date)] snapshot_epoch: view with:"
    echo "    git --no-pager show ${BRANCH}:snapshots/${EXP}_ep${EP}/stats.txt"
    echo "    git --no-pager show ${BRANCH}:snapshots/${EXP}_ep${EP}/index.html > /tmp/ep${EP}.html"
    echo "[$(date)] snapshot_epoch: full render at ${VIZ_DIR}/images/index.html"
else
    echo "[$(date)] snapshot_epoch: ${SNAP_DIR} already recorded, skipping commit"
fi
