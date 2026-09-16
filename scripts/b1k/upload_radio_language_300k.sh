#!/usr/bin/env bash
# Dedicated repository and single writer for the task-name CLIP/FiLM run.
set -euo pipefail
source /tmp/dev/env.sh
cd /tmp/dev/baselines/diffusion_policy
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
RUN=outputs/turning-on-radio-transformer12x512-clipfilm-taskname-bs8960-300k-20260916
LOG=/tmp/dev/logs/dp-radio-clipfilm-taskname-upload-20260916.log
STATUS=/tmp/dev/logs/dp-radio-clipfilm-taskname-upload-20260916.exit
mkdir -p "$(dirname "$LOG")"
if [[ -e "$STATUS" ]]; then
    printf 'Archive the previous exit status before restarting: %s\n' "$STATUS" >&2
    exit 1
fi
trap 'rc=$?; printf "%s\n" "$rc" >"$STATUS.tmp"; mv "$STATUS.tmp" "$STATUS"' EXIT
exec >>"$LOG" 2>&1
read -r trainer_commit <"$RUN/trainer_commit.txt"
taskset -c 122-123 .venv/bin/python -u scripts/b1k/upload_checkpoints.py \
    --run-dir "$RUN" \
    --staging-dir /tmp/dev/hf-staging/dp-radio-clipfilm-taskname-300k-20260916 \
    --repo-id kmy17518/b1k-dp-transformer12x512-turning-on-radio-clipfilm-taskname-20260916 \
    --run-id dp-radio-clipfilm-taskname-300k-20260916 \
    --max-steps 300000 --eval-every 10000 --poll-seconds 30 \
    --metadata policy=DiffusionTransformerHybridImagePolicy --metadata task=turning_on_radio \
    --metadata batch_size=8960 --metadata transformer_layers=12 --metadata embedding_dim=512 \
    --metadata language_conditioning=clip_film --metadata prompt_source=task_name \
    --metadata trainer_commit="$trainer_commit" \
    --metadata wandb_url=https://wandb.ai/kmy17518/b1k-challenge-2026-diffusion-policy/runs/dpradioclipname16
