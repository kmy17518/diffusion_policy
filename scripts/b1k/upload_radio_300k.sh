#!/usr/bin/env bash
# Single writer for this run's eval archive and quota-cleaned latest full checkpoint.
set -euo pipefail
source /tmp/dev/env.sh
cd /tmp/dev/baselines/diffusion_policy
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
LOG=/tmp/dev/logs/dp-radio-upload-20260916.log
STATUS=/tmp/dev/logs/dp-radio-upload-20260916.exit
set +e
taskset -c 122-123 .venv/bin/python -u scripts/b1k/upload_checkpoints.py \
    --run-dir outputs/turning-on-radio-transformer12x512-bs8960-300k-20260916 \
    --staging-dir /tmp/dev/hf-staging/dp-radio-300k-20260916 \
    --repo-id kmy17518/b1k-dp-transformer12x512-turning-on-radio-20260916 \
    --run-id dp-radio-300k-20260916 --max-steps 300000 --eval-every 10000 --poll-seconds 30 \
    --metadata policy=DiffusionTransformerHybridImagePolicy --metadata task=turning_on_radio \
    --metadata batch_size=8960 --metadata transformer_layers=12 --metadata embedding_dim=512 \
    --metadata trainer_commit=0a9269510b35c47f3876e853e871aecc3a22c046 \
    --metadata wandb_url=https://wandb.ai/kmy17518/b1k-challenge-2026-diffusion-policy/runs/dpradio16 \
    >>"$LOG" 2>&1
rc=$?
printf '%s\n' "$rc" >"$STATUS.tmp"
mv "$STATUS.tmp" "$STATUS"
exit "$rc"
