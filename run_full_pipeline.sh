#!/usr/bin/env bash
# Full SLAM SuperPoint+LightGlue pipeline on MAP1, from scratch: regenerate
# pairs (fixed pose matching) -> train SuperPoint -> re-extract SP features
# with the freshly trained SP -> train LightGlue -> inference + dashboard
# visualization over the whole dataset. Safe to run detached so it survives
# an SSH disconnect.
#
# Usage (on the GPU server):
#   cd /home/gpuserver/VOS_dev/slam/glue-factory
#   setsid nohup bash run_full_pipeline.sh [lg_experiment_name] [epochs] [sp_experiment_name] \
#       > "outputs/training/full_pipeline_$(date +%Y%m%d_%H%M).log" 2>&1 < /dev/null &
#   disown
#
# Defaults: lg_experiment_name=lightglue_slam_run_200ep, epochs=200,
#           sp_experiment_name=superpoint_slam_run_200ep
# The same epoch count is used for both SuperPoint and LightGlue.
#
# Optional environment overrides:
#   CLEAN=1           clear outputs/training, outputs/inference,
#                     outputs/gpu_analysis and data/${DATA_DIR}/visualizations
#                     before starting, so the run begins from the official
#                     pretrained weights with nothing carried over. Dataset
#                     images, poses, pseudo-labels and pairs are never touched,
#                     and each path is checked against an allow-list before
#                     deletion. Set CLEAN=0 to keep existing outputs.
#   DATA_DIR         dataset directory under data/            (default MAP1)
#   SP_CONF          SuperPoint config to train               (default .../superpoint_slam_MAP1.yaml)
#   SP_ONLY=1        SuperPoint only: skip steps 4 and 5, and run step 6 with
#                    --matcher none --output all (renders per-pair PNGs +
#                    images/index.html instead of an html-only dashboard).
#                    This is now the default.
#   SKIP_PAIRS=1     skip step 2's pair regeneration. Required when training on
#                    a pre-built subset, otherwise prepare_slam_pairs overwrites
#                    the subset's pairs_train.txt / pairs_val.txt
#   INFER_MAX_PAIRS  cap on consecutive image pairs in step 6  (default 0 = all)
#   INFER_THRESHOLD  step 6 detection_threshold               (default 0.005)
#   SNAPSHOT_EPOCHS  comma-separated epochs at which to snapshot mid-run, e.g.
#                    "10,50". Each spawns snapshot_epoch.sh, which waits for
#                    that epoch's checkpoint, renders 50 pairs, and commits the
#                    gallery to refs/heads/snapshots/<sp_experiment> without
#                    touching the working tree or HEAD. Set to "none" to disable.
#                    Progress: tail outputs/snapshot_<exp>_ep<N>.log
#
# Defaults: SP_ONLY=1, CLEAN=1, epochs=100, threshold=0.005, snapshot at epoch 10.
#
# NOTE on reading loss/detector: the shipped SP config optimizes the descriptor
# only (train.opt_regexp=descriptor) with the detector frozen, so loss/detector
# is a large constant that never trains (7669 on Ouster, 12660 on MAP1). It is
# not a health signal. Judge a run by inference keypoint count vs stock, which
# is what the epoch snapshots and step 6 report.
#
# Check progress any time with (from a NEW ssh session, doesn't need the
# original one to still be open) -- everything this script prints, including
# every line gluefactory.train itself logs, goes to whatever file you
# redirected stdout/stderr to at launch:
#   tail -f outputs/training/full_pipeline_*.log
# gluefactory.train also writes its own copy to each experiment's own log:
#   tail -f "outputs/training/<sp_experiment_name>/log.txt"
#   tail -f "outputs/training/<lg_experiment_name>/log.txt"
#
# Completion marker: outputs/training/${LG_EXP_NAME}.DONE, or
# outputs/training/${SP_EXP_NAME}.DONE when SP_ONLY=1 (touched at the very end,
# after inference/visualization). If that file doesn't exist yet, the pipeline
# is still running or it failed -- check the log for a Python traceback (set -e
# below stops the script on the first failing step).

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

LG_EXP_NAME="${1:-lightglue_slam_run_100ep}"
EPOCHS="${2:-100}"
SP_EXP_NAME="${3:-superpoint_slam_run_100ep}"
DATA_DIR="${DATA_DIR:-MAP1}"
SP_CONF="${SP_CONF:-gluefactory/configs/superpoint_slam_MAP1.yaml}"
SP_ONLY="${SP_ONLY:-1}"
SKIP_PAIRS="${SKIP_PAIRS:-0}"
INFER_MAX_PAIRS="${INFER_MAX_PAIRS:-0}"
INFER_THRESHOLD="${INFER_THRESHOLD:-0.005}"
LG_CONF="gluefactory/configs/superpoint+lightglue_slam_MAP1.yaml"
CLEAN="${CLEAN:-1}"
SNAPSHOT_EPOCHS="${SNAPSHOT_EPOCHS:-10}"
SNAP_PIDS=()

if [ "$SP_ONLY" = "1" ]; then
    echo "=== [$(date)] SP_ONLY=1: steps 4 (feature re-extraction) and 5 (LightGlue) will be skipped ==="
    if [ "$SKIP_PAIRS" != "1" ]; then
        echo "WARNING: SP_ONLY=1 without SKIP_PAIRS=1 will regenerate ${DATA_DIR} pairs," >&2
        echo "         which discards any pre-built pair subset. Set SKIP_PAIRS=1 if that matters." >&2
    fi
fi

echo "=== [$(date)] Pipeline start: sp_experiment=${SP_EXP_NAME} lg_experiment=${LG_EXP_NAME} epochs=${EPOCHS} data=${DATA_DIR} ==="
echo "=== [$(date)] sp_conf=${SP_CONF} threshold=${INFER_THRESHOLD} max_pairs=${INFER_MAX_PAIRS} ==="
echo "=== [$(date)] clean=${CLEAN} snapshot_epochs=${SNAPSHOT_EPOCHS:-none} ==="

# Snapshot watchers are children of this script; make sure they die with it even
# if this script is killed, so an interrupted pipeline never leaves a watcher
# polling for an experiment that will never produce that epoch.
cleanup_watchers() {
    for pid in ${SNAP_PIDS+"${SNAP_PIDS[@]}"}; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup_watchers EXIT INT TERM

# --- Step 0: clear previous run artifacts ----------------------------------
# CLEAN=1 wipes training/inference/visualization outputs so a re-run starts from
# the official pretrained weights with nothing carried over. Dataset images,
# poses, pseudo-labels and pairs are never touched. Paths are resolved and
# checked against an allow-list before anything is deleted, so a bad DATA_DIR
# cannot turn this into an unbounded rm -rf.
if [ "$CLEAN" = "1" ]; then
    echo "=== [$(date)] Step 0/7: clearing previous run artifacts ==="
    REPO_ROOT="$(pwd -P)"
    for target in "outputs/training" "outputs/inference" "outputs/gpu_analysis" \
                  "data/${DATA_DIR}/visualizations"; do
        abs="$(realpath -m "$target")"
        case "$abs" in
            "$REPO_ROOT"/outputs/*|"$REPO_ROOT"/data/*/visualizations)
                if [ -d "$abs" ]; then
                    n="$(find "$abs" -mindepth 1 -maxdepth 1 | wc -l)"
                    echo "  removing ${n} entries from ${abs#"$REPO_ROOT"/}"
                    find "$abs" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
                else
                    echo "  nothing to remove at ${abs#"$REPO_ROOT"/}"
                fi
                ;;
            *)
                echo "ERROR: refusing to delete ${abs}: outside outputs/ and data/*/visualizations/" >&2
                exit 1
                ;;
        esac
    done
elif [ "$CLEAN" != "0" ]; then
    echo "ERROR: CLEAN must be 0 or 1, got '${CLEAN}'" >&2
    exit 1
fi

# --- snapshot watchers ------------------------------------------------------
# Launched before training so they are already polling when the first epoch
# checkpoint lands. Each one waits for its epoch, renders a few frames, and
# commits the result to snapshots/<experiment> without touching the working
# tree -- so a mid-run check never disturbs the training job.
if [ -n "$SNAPSHOT_EPOCHS" ] && [ "$SNAPSHOT_EPOCHS" != "none" ]; then
    IFS=',' read -r -a SNAP_EPOCH_LIST <<< "$SNAPSHOT_EPOCHS"
    for ep in "${SNAP_EPOCH_LIST[@]}"; do
        ep="$(echo "$ep" | tr -d '[:space:]')"
        [ -n "$ep" ] || continue
        echo "=== [$(date)] starting snapshot watcher for epoch ${ep} ==="
        bash snapshot_epoch.sh "$SP_EXP_NAME" "$ep" "$DATA_DIR" \
            "$INFER_THRESHOLD" 50 "snapshots/${SP_EXP_NAME}" 10 \
            > "outputs/snapshot_${SP_EXP_NAME}_ep${ep}.log" 2>&1 &
        SNAP_PIDS+=("$!")
        echo "  watcher pid ${SNAP_PIDS[-1]}, log: outputs/snapshot_${SP_EXP_NAME}_ep${ep}.log"
    done
fi

echo "=== [$(date)] Step 1/7: wait for GPU to be free of other gluefactory.train jobs ==="
# A plain `pgrep -f "gluefactory\.train"` matches any process whose full command
# line merely *contains* that string, so a wrapper shell invoked with the
# training command in its arguments matches itself and this loop never exits.
# Conversely `pgrep -x` cannot be used: a real invocation carries extra
# arguments (experiment name, --conf, overrides) so it is never an exact match
# and every real job would be missed.
#
# So: match the specific substring a real invocation always starts with, then
# drop this script's own process ancestry. That still blocks on genuine
# training while being immune to self-matching.
self_and_ancestors() {
    local p=$$
    while [ -n "$p" ] && [ "$p" -gt 1 ] 2>/dev/null; do
        echo "$p"
        p="$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')"
    done
}
while true; do
    SELF_TREE=" $(self_and_ancestors | tr '\n' ' ')"
    BUSY=""
    for pid in $(pgrep -f "python3 -m gluefactory\.train" 2>/dev/null || true); do
        case "$SELF_TREE" in *" $pid "*) continue ;; esac
        BUSY="${BUSY} ${pid}"
    done
    if [ -z "$BUSY" ]; then
        break
    fi
    echo "  [$(date)] training job(s)${BUSY} still running, waiting 30s..."
    sleep 30
done

if [ "$SKIP_PAIRS" = "1" ]; then
    echo "=== [$(date)] Step 2/7: SKIPPED (SKIP_PAIRS=1), using existing pairs ==="
    echo "Pair counts:"
    wc -l "data/${DATA_DIR}/pairs_train.txt" "data/${DATA_DIR}/pairs_val.txt"
else
    echo "=== [$(date)] Step 2/7: regenerate SLAM pairs (uses the fixed pose-timestamp matching) ==="
    python3 -m gluefactory.scripts.prepare_slam_pairs \
        --data_dir "data/${DATA_DIR}" \
        --min_dist 0.1 --max_dist 2.0 --max_angle 30 --min_overlap 0.1 \
        --max_pairs 10 --split_ratio 0.83 --num_vis 0
    echo "Pair counts:"
    wc -l "data/${DATA_DIR}/pairs_train.txt" "data/${DATA_DIR}/pairs_val.txt"
fi

# Rescale the exponential LR-decay schedule proportionally to EPOCHS. Both
# shipped configs (train.lr_schedule.start=20, exp_div_10=10) were tuned for
# a 50-epoch run (decay starts at 40% through training, then decays 10x
# every exp_div_10 epochs). Keeping those absolute values fixed while epochs
# grows to e.g. 200 would decay the LR to ~0 by epoch ~60-70 and waste the
# rest of training, so scale both by epochs/50. Applied to both SP and LG
# since both configs use the same schedule shape.
LR_START=$(python3 -c "print(max(1, round(20 * ${EPOCHS} / 50)))")
LR_EXP_DIV10=$(python3 -c "print(max(1, round(10 * ${EPOCHS} / 50)))")
echo "LR schedule for ${EPOCHS} epochs: start=${LR_START} exp_div_10=${LR_EXP_DIV10}"

echo "=== [$(date)] Step 3/7: train SuperPoint for ${EPOCHS} epochs (experiment: ${SP_EXP_NAME}) ==="
# Uses the existing pseudo_labels_slam.h5 (per-image consensus keypoints) for
# keypoint/descriptor targets. The SP config inherits the dataset's
# pairs_train.txt/pairs_val.txt for its two_view_pipeline sampling, so it
# benefits from the same pair-count fix as LightGlue -- which is why pair
# regeneration (step 2) normally runs before this step. Note the SP configs
# currently set depth_supervision.do=False, so the detector loss is a plain
# torch.nn.functional.cross_entropy over the 65-class cell targets, not the
# depth-weighted balanced_cross_entropy.
python3 -m gluefactory.train "$SP_EXP_NAME" \
    --conf "$SP_CONF" \
    train.epochs="$EPOCHS" \
    train.lr_schedule.start="$LR_START" \
    train.lr_schedule.exp_div_10="$LR_EXP_DIV10"

SP_CKPT="outputs/training/${SP_EXP_NAME}/checkpoint_best.tar"
if [ ! -f "$SP_CKPT" ]; then
    echo "ERROR: expected SuperPoint checkpoint not found at $SP_CKPT after training" >&2
    exit 1
fi

if [ "$SP_ONLY" = "1" ]; then
    echo "=== [$(date)] Steps 4/7 and 5/7: SKIPPED (SP_ONLY=1) ==="
    echo "  step 4 (extract_slam_features) only feeds the LightGlue targets, and"
    echo "  step 5 trains LightGlue -- neither is needed for a SuperPoint-only run."
else
    echo "=== [$(date)] Step 4/7: re-extract SuperPoint descriptors at consensus keypoints (all images, using the freshly trained SP) ==="
    python3 -m gluefactory.scripts.extract_slam_features \
        --dataset "$DATA_DIR" \
        --pseudo_labels_h5 "data/${DATA_DIR}/exports/pseudo_labels_slam.h5" \
        --weights "$SP_CKPT" \
        --output_h5 "data/${DATA_DIR}/exports/sp_features_slam.h5" \
        --modality rgb

    echo "=== [$(date)] Step 5/7: train LightGlue for ${EPOCHS} epochs (experiment: ${LG_EXP_NAME}) ==="
    python3 -m gluefactory.train "$LG_EXP_NAME" \
        --conf "$LG_CONF" \
        train.epochs="$EPOCHS" \
        train.lr_schedule.start="$LR_START" \
        train.lr_schedule.exp_div_10="$LR_EXP_DIV10"
fi

if [ "$SP_ONLY" = "1" ]; then
    # SuperPoint only: no matcher to run, and --output all so the per-pair
    # matplotlib PNGs and images/index.html are actually written. ("--output
    # html data" renders no PNGs, and "--output data" writes only the JSON.)
    echo "=== [$(date)] Step 6/7: SuperPoint-only inference (--matcher none, rendering PNGs) ==="
    MPLBACKEND=Agg python3 -m gluefactory.scripts.run_inference \
        --backend checkpoint --matcher none \
        --extractor_ckpt "$SP_CKPT" \
        --input "data/${DATA_DIR}/images/rgb" \
        --output all \
        --output_dir "data/${DATA_DIR}/visualizations/sp_${SP_EXP_NAME}" \
        --detection_threshold "$INFER_THRESHOLD" \
        --resize 0 \
        --max_pairs "$INFER_MAX_PAIRS" \
        --log_every 100
    VIS_DIR="data/${DATA_DIR}/visualizations/sp_${SP_EXP_NAME}/images/index.html"
else
    echo "=== [$(date)] Step 6/7: inference + dashboard visualization with the new checkpoints ==="
    # Runs over ALL consecutive image pairs in data/${DATA_DIR}/images/rgb
    # (--max_pairs 0 = no cap), which is ~1812 pairs for the full 1813-image
    # dataset. Per the script's own docstring, skip the per-pair matplotlib PNG
    # plots ("html data" instead of "all") for a batch this size -- they're the
    # main wall-time sink -- and downsample what the dashboard renders so the
    # HTML stays manageable; matches_data.js still has every pair's raw data.
    MPLBACKEND=Agg python3 -m gluefactory.scripts.run_inference \
        --backend checkpoint --matcher lightglue \
        --extractor_ckpt "$SP_CKPT" \
        --matcher_ckpt "outputs/training/${LG_EXP_NAME}/checkpoint_best.tar" \
        --input "data/${DATA_DIR}/images/rgb" \
        --output html data \
        --output_dir "data/${DATA_DIR}/visualizations/sp_lg_${LG_EXP_NAME}" \
        --detection_threshold "$INFER_THRESHOLD" \
        --filter_threshold 0.2 \
        --resize 0 \
        --max_pairs "$INFER_MAX_PAIRS" \
        --downsample_dashboard 5 \
        --save_workers 8 \
        --log_every 100
    VIS_DIR="data/${DATA_DIR}/visualizations/sp_lg_${LG_EXP_NAME}/index.html"
fi

echo "=== [$(date)] ALL DONE ==="
echo "SuperPoint best checkpoint: outputs/training/${SP_EXP_NAME}/checkpoint_best.tar"
echo "SP training log:            outputs/training/${SP_EXP_NAME}/log.txt"
echo "Dashboard:                  ${VIS_DIR}"
if [ -n "$SNAPSHOT_EPOCHS" ] && [ "$SNAPSHOT_EPOCHS" != "none" ]; then
    echo "Snapshot branch:            snapshots/${SP_EXP_NAME}"
    echo "  git --no-pager show snapshots/${SP_EXP_NAME}:snapshots/${SP_EXP_NAME}_ep10/stats.txt"
fi
if [ "$SP_ONLY" = "1" ]; then
    touch "outputs/training/${SP_EXP_NAME}.DONE"
else
    echo "LightGlue best checkpoint:  outputs/training/${LG_EXP_NAME}/checkpoint_best.tar"
    echo "LG training log:            outputs/training/${LG_EXP_NAME}/log.txt"
    touch "outputs/training/${LG_EXP_NAME}.DONE"
fi
