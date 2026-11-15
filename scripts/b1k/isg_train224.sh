#!/usr/bin/env bash
# isg-train224 Diffusion Policy run on the batched replay's native 224 x 224 LeRobot v3 datasets (b1k.md "isg-train224").
#
#   scripts/b1k/isg_train224.sh cache     build ACT's uint8 frame cache (--resize-mode act; native size: no resize). Skip
#                                         it when ACT's launcher already built FRAME_CACHE: the same cache serves both
#   scripts/b1k/isg_train224.sh prepare   exact statistics + the frame-cache sidecar tables (CPU, one process)
#   scripts/b1k/isg_train224.sh train     DDP training (--require-prepared) + the Hugging Face upload monitor beside it;
#                                         resumes from the run directory's latest.pt
#   scripts/b1k/isg_train224.sh upload    the upload monitor alone (resumes its journal)
#   scripts/b1k/isg_train224.sh all       cache, prepare, train
#
# Recipe (dp-relq-{early,late}-piaug, the earlier DP runs' recipe) with the isg-train224 changes: absolute grippers
# (gripper state = q1 + q2 in metres), the sampler not clipped at +-1 (--no-clip-sample; quantile-normalized targets
# exceed it), exact padding-free quantile statistics computed before the run, Pi-style frame epochs (a fresh
# permutation of every training frame per epoch, without replacement), --val-fraction 0.05 (the same episodes as ACT
# / openpi for the seed), pi05's warmup + cosine LR schedule over the whole run, 3 local full checkpoints, online W&B.
#
# Environment (defaults in brackets):
#   DATASET        LeRobot v3 root (required)              FRAME_CACHE [$DATASET-act-frame-cache-224x224]
#   GOAL_FUSION    early | late [early]                    RUN [dp-isg224-relq-$GOAL_FUSION-piaug-ddp4-bs1024-s$SEED]
#   OUT_ROOT       [/tmp/dev/runs/train224]  PY [python]  GPUS [0,1,2,3]  NUM_WORKERS per rank [8]  CACHE_WORKERS [16]
#   MAX_STEPS      [200000]   SAVE [10000] (save = export = validate)   SEED [0]   VAL_FRACTION [0.05]
#   SETTLE_STEPS   [all]: the dataset's own settle window     TASK_NAMES [every task of the dataset]
#   WANDB_ENTITY   [kmy17518]   WANDB_PROJECT [b1k-isg-train224]   WANDB_MODE [online] (needs WANDB_API_KEY)
#   HF_REPO        [kmy17518/$RUN]   HF_PUBLIC [0]   HF_XET [1]   (needs HF_TOKEN with write access)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${DATASET:?set DATASET to the LeRobot v3 root}"
SEED=${SEED:-0}
GOAL_FUSION=${GOAL_FUSION:-early}
FRAME_CACHE=${FRAME_CACHE:-${DATASET%/}-act-frame-cache-224x224}
RUN=${RUN:-dp-isg224-relq-$GOAL_FUSION-piaug-ddp4-bs1024-s$SEED}
OUT_ROOT=${OUT_ROOT:-/tmp/dev/runs/train224}
OUT=$OUT_ROOT/$RUN
STATS=$OUT_ROOT/stats/$RUN.json
PY=${PY:-python}
GPUS=${GPUS:-0,1,2,3}
NPROC=$(( $(tr -cd ',' <<< "$GPUS" | wc -c) + 1 ))
MAX_STEPS=${MAX_STEPS:-200000}
SAVE=${SAVE:-10000}
HF_REPO=${HF_REPO:-kmy17518/$RUN}
case $GOAL_FUSION in
    early) fusion=(--goal-fusion early --goal-stem-init zero) ;;
    late) fusion=(--goal-fusion late --goal-encoder shared_base) ;;
    *) echo "GOAL_FUSION must be early or late" >&2; exit 2 ;;
esac

recipe=(--dataset-path "$DATASET" --episode-split auto --settle-steps "${SETTLE_STEPS:-all}"
    --gripper-state sum --gripper-action absolute --no-task-onehot --regime image --goal-views head
    --goal-source episode_last "${fusion[@]}"
    --variant transformer_hybrid_image --n-layer 12 --n-emb 512 --n-head 8 --conditioning global
    --horizon 16 --n-obs-steps 2 --n-action-steps 8 --image-size 224
    --scheduler ddpm --num-train-timesteps 100 --num-inference-steps 100 --no-clip-sample
    --learning-rate 1e-4 --weight-decay 1e-6 --optimizer adamw --betas 0.9 0.999
    --lr-scheduler pi_cosine --lr-warmup-steps 1000 --lr-schedule-steps "$MAX_STEPS" --lr-decay-ratio 0.1
    --ema-power 0.6667 --grad-clip 1.0
    --frame-cache "$FRAME_CACHE" --resize-mode act --action-representation relative --normalization quantile
    --stats-estimator exact --frame-selection pi --image-augment pi05 --val-fraction "${VAL_FRACTION:-0.05}"
    --seed "$SEED" --batch-size 1024 --max-steps "$MAX_STEPS" --val-every "$SAVE" --stats-file "$STATS")
[[ -z ${TASK_NAMES:-} ]] || recipe+=(--task-names $TASK_NAMES)
runtime=(--loader-batch-size 32 --prefetch-factor 2 --cpu-threads 2 --worker-cpu-threads 1 --episode-cache-size 200
    --matmul-precision high --autocast bf16 --compile default --device cuda --num-workers "${NUM_WORKERS:-8}"
    --save-every "$SAVE" --export-every "$SAVE" --val-batches 16 --val-batch-size 64 --val-goal-batches 2
    --save-total-limit 3 --output-dir "$OUT" --stop-file "$OUT_ROOT/$RUN.stop"
    --wandb-mode "${WANDB_MODE:-online}" --wandb-entity "${WANDB_ENTITY:-kmy17518}"
    --wandb-project "${WANDB_PROJECT:-b1k-isg-train224}" --wandb-name "$RUN" --wandb-id "$RUN")

cache() {
    (cd "$HERE" && "$PY" -u -m diffusion_policy.b1k.frame_cache --dataset-path "$DATASET" --cache-dir "$FRAME_CACHE" \
        --resize-mode act --image-size 224 --workers "${CACHE_WORKERS:-16}" ${TASK_NAMES:+--task-names $TASK_NAMES})
}

prepare() {
    mkdir -p "$OUT_ROOT/stats" "$OUT_ROOT/prepare/$RUN"
    (cd "$HERE" && "$PY" -u -m diffusion_policy.b1k.train "${recipe[@]}" --device cpu --num-workers 0 \
        --output-dir "$OUT_ROOT/prepare/$RUN" --prepare-only)
}

upload() {
    mkdir -p "$OUT/export_queue/eval" "$OUT/export_queue/full" "$OUT_ROOT/upload/$RUN"
    local flags=(--run-dir "$OUT" --staging-dir "$OUT_ROOT/upload/$RUN" --repo-id "$HF_REPO" --run-id "$RUN"
                 --max-steps "$MAX_STEPS" --eval-every "$SAVE" --poll-seconds 60)
    [[ ${HF_PUBLIC:-0} == 1 ]] && flags+=(--public)
    [[ ${HF_XET:-1} == 1 ]] && flags+=(--xet)
    (cd "$HERE" && "$PY" -u scripts/b1k/upload_checkpoints.py "${flags[@]}")
}

train() {
    [[ -f $STATS ]] || { echo "no prepared statistics $STATS: run '$0 prepare' first" >&2; exit 1; }
    mkdir -p "$OUT"
    local resume=()
    [[ ! -e $OUT/latest.pt ]] || resume=(--resume "$OUT/latest.pt")
    rm -f "$OUT_ROOT/$RUN.stop"
    upload >> "$OUT_ROOT/upload/$RUN.log" 2>&1 &
    local uploader=$! rc=0
    echo "[$(date -u +%FT%TZ)] $RUN: training on GPUs $GPUS (${resume[*]:-fresh start}); uploader pid $uploader -> $HF_REPO"
    (cd "$HERE" && CUDA_VISIBLE_DEVICES=$GPUS OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        ARROW_NUM_THREADS=1 PYTORCH_ALLOC_CONF=expandable_segments:True WANDB_CONSOLE=off \
        "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC" \
        -m diffusion_policy.b1k.train "${recipe[@]}" "${runtime[@]}" --require-prepared "${resume[@]}") || rc=$?
    if ((rc == 0)); then
        echo "[$(date -u +%FT%TZ)] $RUN: trainer finished; waiting for the uploader to publish the last checkpoints"
        wait "$uploader" || echo "uploader exited with $?; restart it with '$0 upload'" >&2
    else
        echo "[$(date -u +%FT%TZ)] $RUN: trainer exited with $rc; the uploader (pid $uploader) keeps publishing what is" \
             "queued; '$0 train' resumes both" >&2
    fi
    return "$rc"
}

case ${1:-} in
    cache) cache ;;
    prepare) prepare ;;
    train) train ;;
    upload) upload ;;
    all) cache && prepare && train ;;
    *) sed -n '2,11p' "$0"; exit 2 ;;
esac
