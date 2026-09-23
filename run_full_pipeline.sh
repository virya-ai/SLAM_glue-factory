#!/usr/bin/env bash
# Full SLAM SuperPoint+LightGlue pipeline on MAP1: regenerate pairs (fixed pose
# matching) -> re-extract SP features -> train LightGlue -> inference +
# dashboard visualization. Safe to run detached so it survives an SSH
# disconnect.
#
# Usage (on the GPU server):
#   cd /home/gpuserver/VOS_dev/slam/glue-factory
#   setsid nohup bash run_full_pipeline.sh [experiment_name] [epochs] \
#       > "outputs/training/full_pipeline_$(date +%Y%m%d_%H%M).log" 2>&1 < /dev/null &
#   disown
#
# Defaults: experiment_name=lightglue_slam_run_200ep, epochs=200
#
# Check progress any time with (from a NEW ssh session, doesn't need the
# original one to still be open) -- everything this script prints, including
# every line gluefactory.train itself logs, goes to whatever file you
# redirected stdout/stderr to at launch:
#   tail -f outputs/training/full_pipeline_*.log
# gluefactory.train also writes its own copy to the experiment's own log:
#   tail -f "outputs/training/<experiment_name>/log.txt"
#
# Completion marker: outputs/training/${EXP_NAME}.DONE (touched at the very
# end, after inference/visualization). If that file doesn't exist yet, the
# pipeline is still running or it failed -- check the log for a Python
# traceback (set -e below stops the script on the first failing step).

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

EXP_NAME="${1:-lightglue_slam_run_200ep}"
EPOCHS="${2:-200}"
CONF="gluefactory/configs/superpoint+lightglue_slam_MAP1.yaml"
SP_CKPT="outputs/training/superpoint_slam_run/checkpoint_best.tar"

echo "=== [$(date)] Pipeline start: experiment=${EXP_NAME} epochs=${EPOCHS} ==="

if [ ! -f "$SP_CKPT" ]; then
    echo "ERROR: trained SuperPoint checkpoint not found at $SP_CKPT" >&2
    echo "SuperPoint is already trained and converged cleanly (see outputs/training/superpoint_slam_run/log.txt)." >&2
    echo "This pipeline reuses it rather than retraining; if it's genuinely missing, train it first." >&2
    exit 1
fi

echo "=== [$(date)] Step 1/5: wait for GPU to be free of other gluefactory.train jobs ==="
while pgrep -f "gluefactory\.train" > /dev/null; do
    echo "  [$(date)] another training job is running, waiting 30s..."
    sleep 30
done

echo "=== [$(date)] Step 2/5: regenerate SLAM pairs (uses the fixed pose-timestamp matching) ==="
python3 -m gluefactory.scripts.prepare_slam_pairs \
    --data_dir data/MAP1 \
    --min_dist 0.1 --max_dist 2.0 --max_angle 30 --min_overlap 0.1 \
    --max_pairs 10 --split_ratio 0.83 --num_vis 0
echo "Pair counts:"
wc -l data/MAP1/pairs_train.txt data/MAP1/pairs_val.txt

echo "=== [$(date)] Step 3/5: re-extract SuperPoint descriptors at consensus keypoints (all images) ==="
python3 -m gluefactory.scripts.extract_slam_features \
    --dataset MAP1 \
    --pseudo_labels_h5 data/MAP1/exports/pseudo_labels_slam.h5 \
    --weights "$SP_CKPT" \
    --output_h5 data/MAP1/exports/sp_features_slam.h5 \
    --modality rgb

# Rescale the exponential LR-decay schedule proportionally to EPOCHS. The
# shipped config (train.lr_schedule.start=20, exp_div_10=10) was tuned for a
# 50-epoch run (decay starts at 40% through training, halves roughly every
# ~3 epochs after that). Keeping those absolute values fixed while epochs
# grows to 200 would decay the LR to ~0 by epoch ~60-70 and waste the rest
# of training, so scale both by epochs/50.
LR_START=$(python3 -c "print(max(1, round(20 * ${EPOCHS} / 50)))")
LR_EXP_DIV10=$(python3 -c "print(max(1, round(10 * ${EPOCHS} / 50)))")
echo "LR schedule for ${EPOCHS} epochs: start=${LR_START} exp_div_10=${LR_EXP_DIV10}"

echo "=== [$(date)] Step 4/5: train LightGlue for ${EPOCHS} epochs (experiment: ${EXP_NAME}) ==="
python3 -m gluefactory.train "$EXP_NAME" \
    --conf "$CONF" \
    train.epochs="$EPOCHS" \
    train.lr_schedule.start="$LR_START" \
    train.lr_schedule.exp_div_10="$LR_EXP_DIV10"

echo "=== [$(date)] Step 5/5: inference + dashboard visualization with the new checkpoints ==="
# Runs over ALL consecutive image pairs in data/MAP1/images/rgb (--max_pairs 0
# = no cap), which is ~1812 pairs for the full 1813-image dataset. Per the
# script's own docstring, skip the per-pair matplotlib PNG plots ("html
# data" instead of "all") for a batch this size -- they're the main
# wall-time sink -- and downsample what the dashboard renders so the HTML
# stays manageable; matches_data.js still has every pair's raw data.
MPLBACKEND=Agg python3 -m gluefactory.scripts.run_inference \
    --backend checkpoint --matcher lightglue \
    --extractor_ckpt "$SP_CKPT" \
    --matcher_ckpt "outputs/training/${EXP_NAME}/checkpoint_best.tar" \
    --input data/MAP1/images/rgb \
    --output html data \
    --output_dir "data/MAP1/visualizations/sp_lg_${EXP_NAME}" \
    --max_pairs 0 \
    --downsample_dashboard 5 \
    --save_workers 8 \
    --log_every 100

echo "=== [$(date)] ALL DONE ==="
echo "LightGlue best checkpoint: outputs/training/${EXP_NAME}/checkpoint_best.tar"
echo "Training log:             outputs/training/${EXP_NAME}/log.txt"
echo "Dashboard:                data/MAP1/visualizations/sp_lg_${EXP_NAME}/index.html"
touch "outputs/training/${EXP_NAME}.DONE"
