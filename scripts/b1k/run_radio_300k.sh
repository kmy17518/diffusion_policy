#!/usr/bin/env bash
# Workspace launch recipe: one GPU, 30 CPU cores, 12-layer/512-width image transformer.
#
# Throughput settings (measured on this Blackwell host, see b1k.md "Throughput"):
#   - pixel-exact 96 px frame cache (built/verified below; a no-op once present) so the loader
#     needs 8 workers instead of saturating 24 cores with HEVC decoding,
#   - bf16 autocast + TF32 matmuls + torch.compile of encoder and denoiser, fused AdamW/EMA.
# Overrides: BATCH_SIZE (default 8960), RUN_TAG (default 20260916 = the original run directory,
# which is resumed if it holds latest.pt; any other tag names a fresh run with its own log, exit
# file and W&B run), COMPILE_MODE (default: "reduce-overhead" = CUDA graphs below 4096 samples,
# where launch overhead dominates; "default" above, where CUDA graphs measured slower),
# GPU_UUID (default: this host's GPU 0; one GPU per run; two runs on one GPU would share it and
# neither would reach its measured rate) / CORES, FRAME_CACHE, WANDB_ID. Measured steady state:
# 0.083 s/step at 1024, 0.60 s/step at 8960, excluding the first minute of compilation and the
# periodic checkpoint writes that train.jsonl reports separately as checkpoint_s.
#
# Runs from the checkout that contains this script (main clone or any git worktree), using that
# checkout's .venv and writing its run directory under that checkout's outputs/.
set -euo pipefail
source /tmp/dev/env.sh
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
export CUDA_VISIBLE_DEVICES=${GPU_UUID:-GPU-8a92797c-0df1-b962-f810-3ef2ca82ab80}
if [[ -n "$(nvidia-smi --id "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)" ]]; then
    printf 'Assigned Diffusion Policy GPU is occupied; refusing to start.\n' >&2
    exit 1
fi
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ARROW_NUM_THREADS=1
export WANDB_BASE_URL=https://api.wandb.ai WANDB_MODE=online
BATCH_SIZE=${BATCH_SIZE:-8960}
COMPILE_MODE=${COMPILE_MODE:-$([[ "$BATCH_SIZE" -lt 4096 ]] && echo reduce-overhead || echo default)}
CORES=${CORES:-90-119}
DATASET=/tmp/dev/datasets/2026-challenge-demos
FRAME_CACHE=${FRAME_CACHE:-/tmp/dev/datasets/2026-challenge-demos-frame-cache-96}
RUN_TAG=${RUN_TAG:-20260916}
RUN=outputs/turning-on-radio-transformer12x512-bs${BATCH_SIZE}-300k-${RUN_TAG}
# The original large-batch run keeps its log, exit-status and W&B identity; any other batch size
# or tag gets its own.
IDENT="bs${BATCH_SIZE}-${RUN_TAG}"
if [[ "$IDENT" == bs8960-20260916 ]]; then
    LOG=/tmp/dev/logs/dp-radio-300k-20260916.log
    STATUS=/tmp/dev/logs/dp-radio-300k-20260916.exit
    WANDB_ID=${WANDB_ID:-dpradio16}
else
    LOG=/tmp/dev/logs/dp-radio-300k-${IDENT}.log
    STATUS=/tmp/dev/logs/dp-radio-300k-${IDENT}.exit
    WANDB_ID=${WANDB_ID:-dpradio16-${IDENT}}
fi
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
    --wandb-name "turning-on-radio-transformer12x512-bs${BATCH_SIZE}-300k$([[ "$RUN_TAG" == 20260916 ]] || echo "-${RUN_TAG}")" --wandb-id "$WANDB_ID" \
    "${args[@]}" >>"$LOG" 2>&1
rc=$?
printf '%s\n' "$rc" >"$STATUS.tmp"
mv "$STATUS.tmp" "$STATUS"
exit "$rc"
