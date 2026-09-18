#!/usr/bin/env bash
# Workspace launch recipe: one GPU, 30 CPU cores, 12-layer/512-width image transformer.
#
# Throughput settings (measured on this Blackwell host, see b1k.md "Throughput"):
#   - pixel-exact frame cache at the recipe's image size (built/verified below; a no-op once present)
#     so the loader needs 8 workers instead of saturating 24 cores with HEVC decoding,
#   - bf16 autocast + TF32 matmuls + torch.compile of encoder and denoiser, fused AdamW/EMA.
#
# PRESET (default b1k) selects the model/optimizer recipe:
#   b1k          the runs recorded in b1k_runs.md: 96 px images / 86 px crops, horizon 16, MLP cond
#                encoder, one-hot task id appended to the state, generic AdamW 1e-4 (betas 0.9/0.999,
#                weight decay 1e-6), constant LR, EMA power 2/3, gradient clipping 1.0, batch 8960,
#                300k steps.
#   robocasa365  the RoboCasa365 Diffusion Policy baseline (robocasa-benchmark/diffusion_policy,
#                train_diffusion_transformer_bs192): 256 px images / 224 px crops, horizon 10, 4-layer
#                transformer cond encoder, no one-hot (language is the only task signal), upstream
#                parameter groups with AdamW 1e-4 betas 0.9/0.95, weight decay 1e-3 transformer /
#                1e-6 encoder, cosine LR with 1000 warmup steps sized for 500k steps (their
#                num_epochs x max_train_steps; the released checkpoint is epoch 500 = 250k steps),
#                EMA power 0.75, no gradient clipping, batch 192, 250k steps; language on by default
#                with PROMPT_SOURCE task_description. Adds "robocasa365-" to the identities.
# Overrides: BATCH_SIZE, GRAD_ACCUMULATION (split each optimizer batch into N micro-batches: same
# seeded samples per step, one optimizer step per BATCH_SIZE samples, 1/N the activation memory; adds
# "gaN-" to the identities), LR_SCHEDULE_STEPS (robocasa365 preset: length of the cosine schedule,
# default 500000), MAX_STEPS, IMAGE_SIZE / CROP (square pixels), RUN_TAG (default 20260916 = the
# original run directory, which is resumed if it holds latest.pt; any other tag names a fresh run with
# its own log, exit file and W&B run), COMPILE_MODE (default: "reduce-overhead" = CUDA graphs below
# 4096 samples, where launch overhead dominates; "default" above, where CUDA graphs measured slower),
# GPU_UUID (default: this host's GPU 0; one GPU per run; two runs on one GPU would share it and
# neither would reach its measured rate) / CORES, FRAME_CACHE (default: a cache keyed by IMAGE_SIZE;
# "none" uses the native video loader with 24 workers instead), WANDB_ID. Measured steady state of
# the b1k preset: 0.083 s/step at 1024, 0.60 s/step at 8960, excluding the first minute of
# compilation and the periodic checkpoint writes that train.jsonl reports separately as checkpoint_s.
# LANGUAGE_CONDITIONING (b1k default none; "clip_film" adds frozen CLIP task-prompt FiLM conditioning,
# see b1k.md "Optional CLIP language + FiLM") with PROMPT_SOURCE (task_name, the raw snake_case task
# id, or task_description). Language runs get "clipfilm-<source>-" in their run directory, log, exit
# file and W&B identities, so they never resume an unconditioned run.
# FILM_INIT (default random; "identity" passes --film-init identity: FiLM projections start at zero so
# every conditioned block is initially the identity; adds "identity-" to the identities).
# FILM_RECOMPUTE (default on; "off" passes --no-film-recompute so the FiLM ResNet blocks store their
# activations instead of recomputing them in backward: same math, ~5% faster steps, more memory;
# adds "norecompute-" to the identities so the two variants stay separate runs).
#
# Runs from the checkout that contains this script (main clone or any git worktree), using that
# checkout's .venv and writing its run directory under that checkout's outputs/. The body is a
# function so bash parses the whole file before running it: editing this script while a launched
# run is still executing does not change what that run does at exit.
main() {
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
PRESET=${PRESET:-b1k}
case "$PRESET" in
    b1k)
        PRESET_TAG=''
        IMAGE_SIZE=${IMAGE_SIZE:-96}
        CROP=${CROP:-86}
        BATCH_SIZE=${BATCH_SIZE:-8960}
        MAX_STEPS=${MAX_STEPS:-300000}
        LANGUAGE_CONDITIONING=${LANGUAGE_CONDITIONING:-none}
        PROMPT_SOURCE=${PROMPT_SOURCE:-task_name}
        model_args=(--horizon 16 --n-cond-layers 0 --task-onehot)
        optim_args=(--learning-rate 1e-4 --optimizer adamw --weight-decay 1e-6 --betas 0.9 0.999
                    --lr-scheduler constant --ema-power 0.6666666666666666 --grad-clip 1.0) ;;
    robocasa365)
        PRESET_TAG='robocasa365-'
        IMAGE_SIZE=${IMAGE_SIZE:-256}
        CROP=${CROP:-224}
        BATCH_SIZE=${BATCH_SIZE:-192}
        MAX_STEPS=${MAX_STEPS:-250000}
        LANGUAGE_CONDITIONING=${LANGUAGE_CONDITIONING:-clip_film}
        PROMPT_SOURCE=${PROMPT_SOURCE:-task_description}
        model_args=(--horizon 10 --n-cond-layers 4 --no-task-onehot)
        LR_SCHEDULE_STEPS=${LR_SCHEDULE_STEPS:-500000}
        optim_args=(--learning-rate 1e-4 --optimizer upstream --weight-decay 1e-3 --obs-encoder-weight-decay 1e-6
                    --betas 0.9 0.95 --lr-scheduler cosine --lr-warmup-steps 1000 --lr-schedule-steps "$LR_SCHEDULE_STEPS"
                    --ema-power 0.75 --grad-clip 0) ;;
    *) printf 'PRESET must be b1k or robocasa365\n' >&2
       exit 1 ;;
esac
COMPILE_MODE=${COMPILE_MODE:-$([[ "$BATCH_SIZE" -lt 4096 ]] && echo reduce-overhead || echo default)}
CORES=${CORES:-90-119}
DATASET=/tmp/dev/datasets/2026-challenge-demos
FRAME_CACHE=${FRAME_CACHE:-/tmp/dev/datasets/2026-challenge-demos-frame-cache-${IMAGE_SIZE}}
RUN_TAG=${RUN_TAG:-20260916}
if (( MAX_STEPS % 1000 == 0 )); then STEPS_TAG="$((MAX_STEPS / 1000))k"; else STEPS_TAG="$MAX_STEPS"; fi
case "$LANGUAGE_CONDITIONING/$PROMPT_SOURCE" in
    none/task_name) LANG_TAG=''; lang_args=() ;;
    clip_film/task_name|clip_film/task_description)
        LANG_TAG="clipfilm-${PROMPT_SOURCE//_/}-"
        lang_args=(--language-conditioning clip_film --prompt-source "$PROMPT_SOURCE") ;;
    *) printf 'LANGUAGE_CONDITIONING must be none or clip_film; PROMPT_SOURCE task_name or task_description (clip_film only)\n' >&2
       exit 1 ;;
esac
FILM_INIT=${FILM_INIT:-random}
case "$FILM_INIT" in
    random) ;;
    identity) if [[ "$LANGUAGE_CONDITIONING" != clip_film ]]; then
                  printf 'FILM_INIT=identity requires LANGUAGE_CONDITIONING=clip_film\n' >&2
                  exit 1
              fi
              LANG_TAG="${LANG_TAG}identity-"
              lang_args+=(--film-init identity) ;;
    *) printf 'FILM_INIT must be random or identity\n' >&2
       exit 1 ;;
esac
FILM_RECOMPUTE=${FILM_RECOMPUTE:-on}
case "$FILM_RECOMPUTE" in
    on) ;;
    off) if [[ "$LANGUAGE_CONDITIONING" != clip_film ]]; then
             printf 'FILM_RECOMPUTE=off requires LANGUAGE_CONDITIONING=clip_film\n' >&2
             exit 1
         fi
         LANG_TAG="${LANG_TAG}norecompute-"
         lang_args+=(--no-film-recompute) ;;
    *) printf 'FILM_RECOMPUTE must be on or off\n' >&2
       exit 1 ;;
esac
if [[ "$FRAME_CACHE" == none ]]; then
    loader_args=(--num-workers 24 --loader-batch-size 128)
else
    loader_args=(--num-workers 8 --frame-cache "$FRAME_CACHE")
fi
GRAD_ACCUMULATION=${GRAD_ACCUMULATION:-1}
if (( GRAD_ACCUMULATION < 1 || BATCH_SIZE % GRAD_ACCUMULATION )); then
    printf 'GRAD_ACCUMULATION must be positive and divide BATCH_SIZE\n' >&2
    exit 1
fi
if (( GRAD_ACCUMULATION > 1 )); then
    ACCUM_TAG="ga${GRAD_ACCUMULATION}-"
    if [[ "$FRAME_CACHE" == none ]]; then
        loader_args=(--num-workers 24)   # one worker task per micro-batch; 128-sample chunks need not divide it
    fi
else
    ACCUM_TAG=''
fi
NAME_TAG="${PRESET_TAG}${LANG_TAG}${ACCUM_TAG}bs${BATCH_SIZE}"
RUN=outputs/turning-on-radio-transformer12x512-${NAME_TAG}-${STEPS_TAG}-${RUN_TAG}
# The original large-batch run keeps its log, exit-status and W&B identity; any other preset, batch
# size, tag or language setting gets its own.
IDENT="${NAME_TAG}-${RUN_TAG}"
if [[ "$IDENT" == bs8960-20260916 && "$STEPS_TAG" == 300k ]]; then
    LOG=/tmp/dev/logs/dp-radio-300k-20260916.log
    STATUS=/tmp/dev/logs/dp-radio-300k-20260916.exit
    WANDB_ID=${WANDB_ID:-dpradio16}
else
    LOG=/tmp/dev/logs/dp-radio-${STEPS_TAG}-${IDENT}.log
    STATUS=/tmp/dev/logs/dp-radio-${STEPS_TAG}-${IDENT}.exit
    WANDB_ID=${WANDB_ID:-dpradio16-${IDENT}}
fi
args=()
if [[ -e "$RUN/latest.pt" ]]; then
    args+=(--resume "$RUN/latest.pt")
fi
set +e
if [[ "$FRAME_CACHE" != none ]]; then
    # Decode the task's videos once into resized uint8 frames and check random samples against the
    # native decoder byte for byte. Existing valid entries are skipped, so this is cheap on relaunch.
    CUDA_VISIBLE_DEVICES='' taskset -c "$CORES" .venv/bin/python -u scripts/b1k/build_frame_cache.py \
        --dataset-path "$DATASET" --task-names turning_on_radio --cache-dir "$FRAME_CACHE" \
        --image-size "$IMAGE_SIZE" --workers 28 --verify 256 >>"$LOG" 2>&1 || { printf '%s\n' "$?" >"$STATUS"; exit 1; }
fi
taskset -c "$CORES" .venv/bin/python -u scripts/b1k/train_b1k.py \
    --dataset-path "$DATASET" --task-names turning_on_radio \
    --output-dir "$RUN" --max-steps "$MAX_STEPS" --variant transformer_hybrid_image \
    "${lang_args[@]}" "${model_args[@]}" \
    --n-layer 12 --n-emb 512 --n-head 8 --conditioning global \
    --n-obs-steps 2 --n-action-steps 8 --image-size "$IMAGE_SIZE" --crop-shape "$CROP" "$CROP" \
    --scheduler ddpm --num-train-timesteps 100 --num-inference-steps 100 \
    "${optim_args[@]}" \
    --batch-size "$BATCH_SIZE" --grad-accumulation "$GRAD_ACCUMULATION" "${loader_args[@]}" --prefetch-factor 1 \
    --cpu-threads 2 --worker-cpu-threads 1 --episode-cache-size 200 --device cuda \
    --matmul-precision high --autocast bf16 --compile "$COMPILE_MODE" \
    --save-every 2500 --save-first-step --save-total-limit 3 --export-every 10000 \
    --wandb-mode online --wandb-entity kmy17518 --wandb-project b1k-challenge-2026-diffusion-policy \
    --wandb-name "turning-on-radio-transformer12x512-${NAME_TAG}-${STEPS_TAG}$([[ "$RUN_TAG" == 20260916 ]] || echo "-${RUN_TAG}")" --wandb-id "$WANDB_ID" \
    "${args[@]}" >>"$LOG" 2>&1
rc=$?
printf '%s\n' "$rc" >"$STATUS.tmp"
mv "$STATUS.tmp" "$STATUS"
exit "$rc"
}
main "$@"
