#!/usr/bin/env bash
# Conditioning-regime smoke runs of the 12-layer/512-wide image Diffusion Transformer on the two-task radio mixture
# (5,000 optimizer steps each): one GPU, 30 CPU cores, the b1k preset of scripts/b1k/run_radio_300k.sh (96 px / 86 px
# crops, horizon 16, batch 8,960, bf16 autocast + TF32 + torch.compile, 96 px frame cache).
#
# Dataset: /tmp/dev/datasets/2026-challenge-demos-radio-navpickup-goal (merge of the nav-goal and pickup-goal
# skill-segment datasets; 400 episodes, tasks turning_on_radio-navigate_to_radio and turning_on_radio-pick_up_radio).
# Language = the task name (--prompt-source task_name); goal image = the last frame of the episode's head camera.
#
# DP_CONDITION selects the run and, by default, the checkout whose code trains it:
#   vanilla                the `my` branch recipe (checkout /tmp/dev/baselines/diffusion_policy): no language, no goal;
#                          NOTE `my` always appends the one-hot task id to the state, so on two tasks this is the
#                          task-ID-conditioned baseline, not the plan's strict N (use `none` for that)
#   none                   strict N from this checkout: --regime none --no-task-onehot
#   language               RoboCasa365-style CLIP->ResNet FiLM from the lang_optimized_robocasa365 checkout
#                          (/tmp/dev/baselines/diffusion_policy_lang_goal), b1k preset: --language-conditioning clip_film
#                          --prompt-source task_name --no-task-onehot (the branch's full RoboCasa365 recipe preset --
#                          256 px, horizon 10, transformer cond encoder, cosine LR, batch 192 -- is not used here)
#   image-early | image-late                     --regime image, goal fusion early | late (this checkout)
#   image_language-early | image_language-late   --regime image_language with clip_film language (this checkout)
# Overrides: DP_CHECKOUT, DP_GPU_UUID (default GPU 1), DP_CORES (default 30-59), DP_BATCH_SIZE (8960), DP_MAX_STEPS
# (5000), DP_RUN_TAG (20260920), DP_COMPILE_MODE (default: reduce-overhead below 4096 samples, else default),
# DP_DATASET, DP_FRAME_CACHE, DP_WANDB_ID, DP_WANDB_MODE (online). Extra arguments are passed to train_b1k.py.
main() {
set -euo pipefail
source /tmp/dev/env.sh
CONDITION=${DP_CONDITION:?set DP_CONDITION (vanilla|none|language|image-early|image-late|image_language-early|image_language-late)}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
# The lang/goal branches expose the b1k preset's implicit optimizer settings as flags; `my` has them built in.
preset_args=(--n-cond-layers 0 --lr-scheduler constant --ema-power 0.6666666666666666 --grad-clip 1.0)
case "$CONDITION" in
    vanilla)  CHECKOUT=${DP_CHECKOUT:-/tmp/dev/baselines/diffusion_policy}; cond_args=() ;;
    none)     CHECKOUT=${DP_CHECKOUT:-$HERE}; cond_args=(--regime none --no-task-onehot "${preset_args[@]}") ;;
    language) CHECKOUT=${DP_CHECKOUT:-/tmp/dev/baselines/diffusion_policy_lang_goal}
              cond_args=(--language-conditioning clip_film --prompt-source task_name --no-task-onehot "${preset_args[@]}") ;;
    image-early|image-late)
              CHECKOUT=${DP_CHECKOUT:-$HERE}
              cond_args=(--regime image --goal-fusion "${CONDITION#image-}" --goal-views head --goal-source episode_last
                         --no-task-onehot "${preset_args[@]}") ;;
    image_language-early|image_language-late)
              CHECKOUT=${DP_CHECKOUT:-$HERE}
              cond_args=(--regime image_language --language-conditioning clip_film --prompt-source task_name
                         --goal-fusion "${CONDITION#image_language-}" --goal-views head --goal-source episode_last
                         --no-task-onehot "${preset_args[@]}") ;;
    *) printf 'Unknown DP_CONDITION %s\n' "$CONDITION" >&2; exit 2 ;;
esac
cd "$CHECKOUT"
export CUDA_VISIBLE_DEVICES=${DP_GPU_UUID:-GPU-82d44829-7cec-7d3c-9918-6dc1321320d4}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ARROW_NUM_THREADS=1
export WANDB_BASE_URL=https://api.wandb.ai
BATCH_SIZE=${DP_BATCH_SIZE:-8960}
MAX_STEPS=${DP_MAX_STEPS:-5000}
RUN_TAG=${DP_RUN_TAG:-20260920}
COMPILE_MODE=${DP_COMPILE_MODE:-$([[ "$BATCH_SIZE" -lt 4096 ]] && echo reduce-overhead || echo default)}
CORES=${DP_CORES:-30-59}
DATASET=${DP_DATASET:-/tmp/dev/datasets/2026-challenge-demos-radio-navpickup-goal}
FRAME_CACHE=${DP_FRAME_CACHE:-/tmp/dev/datasets/2026-challenge-demos-frame-cache-96}
WANDB_MODE_ARG=${DP_WANDB_MODE:-online}
TASKS=(turning_on_radio-navigate_to_radio turning_on_radio-pick_up_radio)
STEM=navpickup-dp-transformer12x512-${CONDITION}-bs${BATCH_SIZE}-$((MAX_STEPS / 1000))k-${RUN_TAG}
RUN=outputs/$STEM
LOG=/tmp/dev/logs/${STEM}.log
STATUS=/tmp/dev/logs/${STEM}.exit
WANDB_ID=${DP_WANDB_ID:-dpnp-${CONDITION}-${RUN_TAG}}
mkdir -p "$(dirname "$LOG")"
if [[ -e "$STATUS" ]]; then
    printf 'Archive the previous exit status before restarting: %s\n' "$STATUS" >&2
    exit 1
fi
trap 'rc=$?; printf "%s\n" "$rc" >"$STATUS.tmp"; mv "$STATUS.tmp" "$STATUS"' EXIT
exec >>"$LOG" 2>&1
if [[ -n "$(nvidia-smi --id "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)" ]]; then
    printf 'Assigned GPU %s is occupied; refusing to start.\n' "$CUDA_VISIBLE_DEVICES" >&2
    exit 1
fi
mkdir -p "$RUN"
if [[ ! -e "$RUN/trainer_commit.txt" ]]; then
    { git rev-parse HEAD; git rev-parse --abbrev-ref HEAD; } >"$RUN/trainer_commit.txt"
fi
printf '[%s] condition=%s checkout=%s commit=%s gpu=%s cores=%s\n' "$(date -u +%FT%TZ)" "$CONDITION" "$CHECKOUT" \
    "$(git rev-parse --short HEAD)" "$CUDA_VISIBLE_DEVICES" "$CORES"
# Cache entries for the merged root are the challenge-demo entries (same files); this validates them (no rebuild).
CUDA_VISIBLE_DEVICES='' taskset -c "$CORES" .venv/bin/python -u scripts/b1k/build_frame_cache.py \
    --dataset-path "$DATASET" --task-names "${TASKS[@]}" --cache-dir "$FRAME_CACHE" \
    --image-size 96 --workers 28 --verify 32
args=()
if [[ -e "$RUN/latest.pt" ]]; then
    args+=(--resume "$RUN/latest.pt")
fi
taskset -c "$CORES" .venv/bin/python -u scripts/b1k/train_b1k.py \
    --dataset-path "$DATASET" --task-names "${TASKS[@]}" \
    --output-dir "$RUN" --max-steps "$MAX_STEPS" --variant transformer_hybrid_image \
    "${cond_args[@]}" \
    --n-layer 12 --n-emb 512 --n-head 8 --conditioning global \
    --horizon 16 --n-obs-steps 2 --n-action-steps 8 --image-size 96 --crop-shape 86 86 \
    --scheduler ddpm --num-train-timesteps 100 --num-inference-steps 100 \
    --learning-rate 1e-4 --optimizer adamw --weight-decay 1e-6 --betas 0.9 0.999 \
    --batch-size "$BATCH_SIZE" --num-workers 8 --prefetch-factor 1 \
    --cpu-threads 2 --worker-cpu-threads 1 --episode-cache-size 400 --device cuda \
    --frame-cache "$FRAME_CACHE" \
    --matmul-precision high --autocast bf16 --compile "$COMPILE_MODE" \
    --save-every 2500 --save-first-step --save-total-limit 3 --export-every 5000 \
    --wandb-mode "$WANDB_MODE_ARG" --wandb-entity kmy17518 --wandb-project b1k-challenge-2026-diffusion-policy \
    --wandb-name "$STEM" --wandb-id "$WANDB_ID" \
    "${args[@]}" "$@"
}
main "$@"
