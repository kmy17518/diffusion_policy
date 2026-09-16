#!/usr/bin/env bash
# Task-name CLIP/FiLM run; defaults match the documented unconditioned radio run.
set -euo pipefail
source /tmp/dev/env.sh
cd /tmp/dev/baselines/diffusion_policy
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-GPU-2aa27438-4ef0-fdca-070a-4138fa04301a}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ARROW_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_BASE_URL=https://api.wandb.ai WANDB_MODE=online
RUN=${B1K_RUN_DIR:-outputs/turning-on-radio-transformer12x512-clipfilm-taskname-bs8960-300k-20260916}
LOG=${B1K_LOG:-/tmp/dev/logs/dp-radio-clipfilm-taskname-300k-20260916.log}
STATUS=${B1K_STATUS:-/tmp/dev/logs/dp-radio-clipfilm-taskname-300k-20260916.exit}
mkdir -p "$(dirname "$LOG")"
if [[ -e "$STATUS" ]]; then
    printf 'Archive the previous exit status before restarting: %s\n' "$STATUS" >&2
    exit 1
fi
trap 'rc=$?; printf "%s\n" "$rc" >"$STATUS.tmp"; mv "$STATUS.tmp" "$STATUS"' EXIT
exec >>"$LOG" 2>&1
if [[ -n "$(nvidia-smi --id "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)" ]]; then
    printf 'Assigned Diffusion Policy GPU is occupied; refusing to start.\n' >&2
    exit 1
fi
mkdir -p "$RUN"
if [[ ! -e "$RUN/trainer_commit.txt" ]]; then
    git rev-parse HEAD >"$RUN/trainer_commit.txt"
fi
args=()
if [[ -e "$RUN/latest.pt" ]]; then
    args+=(--resume "$RUN/latest.pt")
fi
taskset -c 90-119 .venv/bin/python -u scripts/b1k/train_b1k.py \
    --dataset-path /tmp/dev/datasets/2026-challenge-demos --task-names turning_on_radio \
    --output-dir "$RUN" --max-steps 300000 --variant transformer_hybrid_image \
    --language-conditioning clip_film --prompt-source task_name \
    --n-layer 12 --n-emb 512 --n-head 8 --conditioning global \
    --horizon 16 --n-obs-steps 2 --n-action-steps 8 --image-size 96 --crop-shape 86 86 \
    --scheduler ddpm --num-train-timesteps 100 --num-inference-steps 100 \
    --learning-rate 1e-4 --optimizer adamw --weight-decay 1e-6 --betas 0.9 0.999 \
    --batch-size 8960 --loader-batch-size 128 --num-workers 24 --prefetch-factor 1 \
    --cpu-threads 2 --worker-cpu-threads 1 --episode-cache-size 200 --device cuda \
    --save-every 2500 --save-first-step --save-total-limit 3 --export-every 10000 \
    --wandb-mode online --wandb-entity kmy17518 --wandb-project b1k-challenge-2026-diffusion-policy \
    --wandb-name turning-on-radio-transformer12x512-clipfilm-taskname-bs8960-300k \
    --wandb-id dpradioclipname16 \
    "${args[@]}" "$@"
