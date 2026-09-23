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
# Check progress any time with (from a NEW ssh session, doesn't need the
# original one to still be open) -- everything this script prints, including
# every line gluefactory.train itself logs, goes to whatever file you
# redirected stdout/stderr to at launch:
#   tail -f outputs/training/full_pipeline_*.log
# gluefactory.train also writes its own copy to each experiment's own log:
#   tail -f "outputs/training/<sp_experiment_name>/log.txt"
#   tail -f "outputs/training/<lg_experiment_name>/log.txt"
#
# Completion marker: outputs/training/${LG_EXP_NAME}.DONE (touched at the
# very end, after inference/visualization). If that file doesn't exist yet,
# the pipeline is still running or it failed -- check the log for a Python
# traceback (set -e below stops the script on the first failing step).

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

LG_EXP_NAME="${1:-lightglue_slam_run_200ep}"
EPOCHS="${2:-200}"
SP_EXP_NAME="${3:-superpoint_slam_run_200ep}"
SP_CONF="gluefactory/configs/superpoint_slam_MAP1.yaml"
LG_CONF="gluefactory/configs/superpoint+lightglue_slam_MAP1.yaml"

echo "=== [$(date)] Pipeline start: sp_experiment=${SP_EXP_NAME} lg_experiment=${LG_EXP_NAME} epochs=${EPOCHS} ==="

echo "=== [$(date)] Step 1/6: wait for GPU to be free of other gluefactory.train jobs ==="
while pgrep -f "gluefactory\.train" > /dev/null; do
    echo "  [$(date)] another training job is running, waiting 30s..."
    sleep 30
done

echo "=== [$(date)] Step 2/6: regenerate SLAM pairs (uses the fixed pose-timestamp matching) ==="
python3 -m gluefactory.scripts.prepare_slam_pairs \
    --data_dir data/MAP1 \
    --min_dist 0.1 --max_dist 2.0 --max_angle 30 --min_overlap 0.1 \
    --max_pairs 10 --split_ratio 0.83 --num_vis 0
echo "Pair counts:"
wc -l data/MAP1/pairs_train.txt data/MAP1/pairs_val.txt

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

echo "=== [$(date)] Step 3/6: train SuperPoint for ${EPOCHS} epochs (experiment: ${SP_EXP_NAME}) ==="
# Uses the existing pseudo_labels_slam.h5 (per-image consensus keypoints) for
# keypoint/descriptor targets. The SP config also inherits the dataset's
# default pairs_train.txt/pairs_val.txt (used for its cross-frame
# depth-consistency supervision, depth_supervision.do=True), so it benefits
# from the same pair-count fix as LightGlue -- this is why pair regeneration
# (step 2) runs before this step.
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

echo "=== [$(date)] Step 4/6: re-extract SuperPoint descriptors at consensus keypoints (all images, using the freshly trained SP) ==="
python3 -m gluefactory.scripts.extract_slam_features \
    --dataset MAP1 \
    --pseudo_labels_h5 data/MAP1/exports/pseudo_labels_slam.h5 \
    --weights "$SP_CKPT" \
    --output_h5 data/MAP1/exports/sp_features_slam.h5 \
    --modality rgb

echo "=== [$(date)] Step 5/6: train LightGlue for ${EPOCHS} epochs (experiment: ${LG_EXP_NAME}) ==="
python3 -m gluefactory.train "$LG_EXP_NAME" \
    --conf "$LG_CONF" \
    train.epochs="$EPOCHS" \
    train.lr_schedule.start="$LR_START" \
    train.lr_schedule.exp_div_10="$LR_EXP_DIV10"

echo "=== [$(date)] Step 6/6: inference + dashboard visualization with the new checkpoints ==="
# Runs over ALL consecutive image pairs in data/MAP1/images/rgb (--max_pairs 0
# = no cap), which is ~1812 pairs for the full 1813-image dataset. Per the
# script's own docstring, skip the per-pair matplotlib PNG plots ("html
# data" instead of "all") for a batch this size -- they're the main
# wall-time sink -- and downsample what the dashboard renders so the HTML
# stays manageable; matches_data.js still has every pair's raw data.
MPLBACKEND=Agg python3 -m gluefactory.scripts.run_inference \
    --backend checkpoint --matcher lightglue \
    --extractor_ckpt "$SP_CKPT" \
    --matcher_ckpt "outputs/training/${LG_EXP_NAME}/checkpoint_best.tar" \
    --input data/MAP1/images/rgb \
    --output html data \
    --output_dir "data/MAP1/visualizations/sp_lg_${LG_EXP_NAME}" \
    --max_pairs 0 \
    --downsample_dashboard 5 \
    --save_workers 8 \
    --log_every 100

echo "=== [$(date)] ALL DONE ==="
echo "SuperPoint best checkpoint: outputs/training/${SP_EXP_NAME}/checkpoint_best.tar"
echo "LightGlue best checkpoint:  outputs/training/${LG_EXP_NAME}/checkpoint_best.tar"
echo "SP training log:            outputs/training/${SP_EXP_NAME}/log.txt"
echo "LG training log:            outputs/training/${LG_EXP_NAME}/log.txt"
echo "Dashboard:                  data/MAP1/visualizations/sp_lg_${LG_EXP_NAME}/index.html"
touch "outputs/training/${LG_EXP_NAME}.DONE"
