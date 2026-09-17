#!/usr/bin/env bash
# Workspace launch recipe: one GPU, 30 CPU cores, 12-layer/512-width image transformer.
#
# Throughput settings (measured on this Blackwell host, see b1k.md "Throughput"):
#   - pixel-exact 96 px frame cache (built/verified below; a no-op once present) so the loader
#     needs 8 workers instead of saturating 24 cores with HEVC decoding,
#   - bf16 autocast + TF32 matmuls + torch.compile of encoder and denoiser, fused AdamW/EMA.
# BATCH_SIZE=1024 COMPILE_MODE=reduce-overhead is the measured best for the small-batch target;
# the default (8960, compile mode "default") is the large-batch run.
set -euo pipefail
source /tmp/dev/env.sh
cd /tmp/dev/baselines/diffusion_policy
export CUDA_VISIBLE_DEVICES=GPU-2f892c97-af70-7c60-d2fa-456c65bd90ce
if [[ -n "$(nvidia-smi --id "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)" ]]; then
    printf 'Assigned Diffusion Policy GPU is occupied; refusing to start.\n' >&2
    exit 1
fi
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ARROW_NUM_THREADS=1
export WANDB_BASE_URL=https://api.wandb.ai WANDB_MODE=online
BATCH_SIZE=${BATCH_SIZE:-8960}
COMPILE_MODE=${COMPILE_MODE:-default}
CORES=${CORES:-90-119}
DATASET=/tmp/dev/datasets/2026-challenge-demos
FRAME_CACHE=${FRAME_CACHE:-/tmp/dev/datasets/2026-challenge-demos-frame-cache-96}
RUN=outputs/turning-on-radio-transformer12x512-bs${BATCH_SIZE}-300k-20260916
LOG=/tmp/dev/logs/dp-radio-300k-20260916.log
STATUS=/tmp/dev/logs/dp-radio-300k-20260916.exit
# The original large-batch run keeps its W&B identity; other batch sizes get their own run ID.
WANDB_ID=${WANDB_ID:-$([[ "$BATCH_SIZE" == 8960 ]] && echo dpradio16 || echo "dpradio16bs${BATCH_SIZE}")}
args=()
if [[ -e "$RUN/latest.pt" ]]; then
    args+=(--resume "$RUN/latest.pt")
fi
set +e
# Decode the task's videos once into resized uint8 frames and check random samples against the
# native decoder byte for byte. Existing valid entries are skipped, so this is cheap on relaunch.
CUDA_VISIBLE_DEVICES='' taskset -c "$CORES" .venv/bin/python -u scripts/b1k/build_frame_cache.py \
    --dataset-path "$DATASET" --task-names turning_on_radio --cache-dir "$FRAME_CACHE" \
    --image-size 96 --workers 28 --verify 256 >>"$LOG" 2>&1 || { printf '%s\n' "$?" >"$STATUS"; exit 1; }
taskset -c "$CORES" .venv/bin/python -u scripts/b1k/train_b1k.py \
    --dataset-path "$DATASET" --task-names turning_on_radio \
    --output-dir "$RUN" --max-steps 300000 --variant transformer_hybrid_image \
    --n-layer 12 --n-emb 512 --n-head 8 --conditioning global \
    --horizon 16 --n-obs-steps 2 --n-action-steps 8 --image-size 96 --crop-shape 86 86 \
    --scheduler ddpm --num-train-timesteps 100 --num-inference-steps 100 \
    --learning-rate 1e-4 --optimizer adamw --weight-decay 1e-6 --betas 0.9 0.999 \
    --batch-size "$BATCH_SIZE" --num-workers 8 --prefetch-factor 1 \
    --cpu-threads 2 --worker-cpu-threads 1 --episode-cache-size 200 --device cuda \
    --frame-cache "$FRAME_CACHE" \
    --matmul-precision high --autocast bf16 --compile "$COMPILE_MODE" \
    --save-every 2500 --save-first-step --save-total-limit 3 --export-every 10000 \
    --wandb-mode online --wandb-entity kmy17518 --wandb-project b1k-challenge-2026-diffusion-policy \
    --wandb-name turning-on-radio-transformer12x512-bs${BATCH_SIZE}-300k --wandb-id "$WANDB_ID" \
    "${args[@]}" >>"$LOG" 2>&1
rc=$?
printf '%s\n' "$rc" >"$STATUS.tmp"
mv "$STATUS.tmp" "$STATUS"
exit "$rc"
