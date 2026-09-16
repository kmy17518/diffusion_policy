#!/usr/bin/env bash
# Workspace launch recipe: one GPU, 30 CPU cores, 12-layer/512-width image transformer.
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
RUN=outputs/turning-on-radio-transformer12x512-bs8960-300k-20260916
LOG=/tmp/dev/logs/dp-radio-300k-20260916.log
STATUS=/tmp/dev/logs/dp-radio-300k-20260916.exit
args=()
if [[ -e "$RUN/latest.pt" ]]; then
    args+=(--resume "$RUN/latest.pt")
fi
set +e
taskset -c 90-119 .venv/bin/python -u scripts/b1k/train_b1k.py \
    --dataset-path /tmp/dev/datasets/2026-challenge-demos --task-names turning_on_radio \
    --output-dir "$RUN" --max-steps 300000 --variant transformer_hybrid_image \
    --n-layer 12 --n-emb 512 --n-head 8 --conditioning global \
    --horizon 16 --n-obs-steps 2 --n-action-steps 8 --image-size 96 --crop-shape 86 86 \
    --scheduler ddpm --num-train-timesteps 100 --num-inference-steps 100 \
    --learning-rate 1e-4 --optimizer adamw --weight-decay 1e-6 --betas 0.9 0.999 \
    --batch-size 8960 --loader-batch-size 128 --num-workers 24 --prefetch-factor 1 \
    --cpu-threads 2 --worker-cpu-threads 1 --episode-cache-size 200 --device cuda \
    --save-every 2500 --save-first-step --save-total-limit 3 --export-every 10000 \
    --wandb-mode online --wandb-entity kmy17518 --wandb-project b1k-challenge-2026-diffusion-policy \
    --wandb-name turning-on-radio-transformer12x512-bs8960-300k --wandb-id dpradio16 \
    "${args[@]}" >>"$LOG" 2>&1
rc=$?
printf '%s\n' "$rc" >"$STATUS.tmp"
mv "$STATUS.tmp" "$STATUS"
exit "$rc"
