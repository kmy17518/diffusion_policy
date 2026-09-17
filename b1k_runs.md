# Transformer Diffusion Policy radio run — 2026-09-16

## Task-name CLIP/FiLM smoke run on the optimized trainer — 2026-09-17 (branch `diffusion_policy_lang_goal`)

After merging the language branch into the throughput work, the radio recipe was launched with `--language-conditioning clip_film --prompt-source task_name` (prompt = the literal `turning_on_radio`), physical batch **8,960**, frame cache, bf16 autocast, TF32, `torch.compile default`, math attention — i.e. the optimized recipe below plus language:

```bash
source /tmp/dev/env.sh
tmux -L b1k-act-dp new-session -d -s dp-radio-lang-goal-train \
  'LANGUAGE_CONDITIONING=clip_film PROMPT_SOURCE=task_name RUN_TAG=lang-goal-20260917 \
   bash /tmp/dev/baselines/diffusion_policy_lang_goal/scripts/b1k/run_radio_300k.sh'
```

- Run directory `outputs/turning-on-radio-transformer12x512-clipfilm-taskname-bs8960-300k-lang-goal-20260917/` (worktree), log `/tmp/dev/logs/dp-radio-300k-clipfilm-taskname-bs8960-lang-goal-20260917.log`, W&B run `dpradio16-clipfilm-taskname-bs8960-lang-goal-20260917` (same project), GPU 0 of this host, cores 90-119, trainer commit `6018b73`.
- First 134 steps: **0.671 s/step** (compute 0.669, data wait 1.4 ms; 13.4k samples/s) versus 0.60 s/step unconditioned; peak allocated **128.6 GiB** versus 157 GiB unconditioned and 189 GiB for the FP32 eager language run. Step 1 (compilation + first batch) took 171 s.
- Losses: steps 1-5 **1.2110 / 1.3594 / 1.1107 / 0.9696 / 0.9393** (unconditioned optimized run: 1.1988 / 1.2759 / 1.0615 / 0.9384 / 0.9092); mean over steps 6-60 **0.3908** versus 0.3752; all finite, max clipped gradient norm 1.69. The step-1 full checkpoint carries the `language` cache (`openai/clip-vit-large-patch14` @ `32bd6428…`, prompt `turning_on_radio`, `[1, 768]`) and 48 FiLM tensors.
- Controlled A/B at batch 1,024, 150 steps, same seed, same flags (`/tmp/dev/audits/dp-lang-goal-20260917/ab-bs1024-{none,clipfilm-taskname}/train.jsonl`): none **0.0808 s/step**, step-1 loss 1.1985, mean loss steps 101-150 0.1629; clip_film/task_name **0.0942 s/step**, 1.2114, 0.1739. Peak memory 19.3 vs 16.2 GiB.
- This is a throughput/loss smoke run launched as a full 300k-step recipe; no uploader was started. Stop it with `tmux -L b1k-act-dp kill-session -t dp-radio-lang-goal-train` if it is not meant to continue.

## Task-name CLIP/FiLM run

**Stopped by user on 2026-09-17 at 05:15 UTC.** The trainer stopped after recorded step **3,675**; the latest resumable checkpoint is **step 2,500**. Trainer, uploader, scheduled health reviews and failure watcher are stopped. Checkpoints and upload journals remain intact. The historical launch/monitoring statements below describe the earlier running state; do not restart this run without a new request.

The language-conditioned reproduction uses the same radio subset, **300,000 steps**, physical batch **8,960**, 12-layer/512-width transformer, FP32 optimizer, image/crop sizes, CPU affinity and checkpoint schedule below, with `--language-conditioning clip_film --prompt-source task_name`. The original one-hot task input remains. Frozen CLIP ViT-L/14 projected embeddings condition every camera's ResNet residual blocks through FiLM and are included in the observation features, as in the RoboCasa baseline. Conditioned GroupNorm blocks use activation recomputation to retain the original physical batch; this is not gradient accumulation or mixed precision.

The new run has separate local, Hugging Face and W&B identities; it never resumes or replaces the original run:

- Run directory: `outputs/turning-on-radio-transformer12x512-clipfilm-taskname-bs8960-300k-20260916/`.
- Private Hugging Face destination: https://huggingface.co/kmy17518/b1k-dp-transformer12x512-turning-on-radio-clipfilm-taskname-20260916
- W&B project is unchanged; new run: https://wandb.ai/kmy17518/b1k-challenge-2026-diffusion-policy/runs/dpradioclipname16
- W&B experiment: `turning-on-radio-transformer12x512-clipfilm-taskname-bs8960-300k`.
- GPU 3 on the current node: `GPU-2aa27438-4ef0-fdca-070a-4138fa04301a` (the historical GPU UUID below belongs to the prior node).
- Trainer log/exit: `/tmp/dev/logs/dp-radio-clipfilm-taskname-300k-20260916.{log,exit}`.
- Uploader log/exit: `/tmp/dev/logs/dp-radio-clipfilm-taskname-upload-20260916.{log,exit}`.
- Durable uploader journal: `/tmp/dev/hf-staging/dp-radio-clipfilm-taskname-300k-20260916/`.

Launch the trainer first, then wait for its first `latest.pt` checkpoint before starting the uploader. Launch provenance lives in the sibling `<run-directory>.trainer_commit.txt` file. The new scripts reject occupied GPUs and existing exit files; preserve the journal and archive an old exit file before a deliberate restart. Shell exit traps also record startup failures. Expandable CUDA segments are enabled to reduce allocator fragmentation without changing model math.

```bash
source /tmp/dev/env.sh
tmux -L b1k-act-dp-language new-session -d -s dp-radio-language-train \
  'bash /tmp/dev/baselines/diffusion_policy/scripts/b1k/run_radio_language_300k.sh'
tmux -L b1k-act-dp-language new-session -d -s dp-radio-language-upload \
  'bash /tmp/dev/baselines/diffusion_policy/scripts/b1k/upload_radio_language_300k.sh'
CUDA_VISIBLE_DEVICES='' /tmp/dev/baselines/act/.venv/bin/python \
  /tmp/dev/scripts/act-dp-language-status.py
```

Qualification passed **16 fresh-data optimizer steps at batch 8,960**, then full-checkpoint resume through **step 20 with Hugging Face/transformers offline**. All **200 episodes** were selected with no episode cap and exact selected-frame normalization. Peak allocated memory was **189.36 GiB**; median steady compute was **2.610 s/step**, plus approximately 8–9 s of native-video data wait. Three local full checkpoints and two eval exports were verified. Real step-16 EMA eval serving passed health, eight websocket action requests, reset, replanning, batching and separate-client checks with dataset/CLIP access denied. All 100 task-name and description embeddings matched ACT exactly; the five long descriptions were chunked without truncation.

The 300,000-step trainer was launched on **2026-09-16 at 18:11 UTC** from commit `81839b5` using the new identities above. It is a fresh run, not the qualification checkpoint. The uploader published `resume/step-00000001.pt` to the new private repo; local SHA-256 and remote LFS SHA-256 matched (`9bdc0c34356379fefa5de3c4e0f223c61a75f23343d7e095a60241fbaf6b9014`). The original checkpoint repo was untouched. A session monitor checks both trainers/uploaders every 30 seconds, and a durable 10-minute health review checks loss, timings and publication status. Training is **in progress**, not complete; no simulator success rate is claimed. Local qualification and monitoring artifacts use `/tmp/dev/audits/act-dp-language-20260916/`. Tmux survives client disconnects, not machine/container termination.

## Controlled short initialization comparison — 2026-09-17

A separate diagnostic uses the full 12-layer/512-width DP architecture above with **1,000 steps per arm**, physical batch **512**, and seed **42**. Three arms share identical initial parameters/buffers: unconditioned baseline, randomly initialized FiLM, and identity-initialized FiLM (`beta=gamma=0`). All three camera encoders and transformer tensors are explicitly mapped. In both language arms, added language columns of the observation projection start at zero; original columns are copied in their actual feature order. This isolates FiLM initialization instead of conflating it with the added direct language input.

Each arm receives the same minibatch and paired crops/dropout/diffusion noise/timesteps. Every tenth sorted episode is held out: **180 train / 20 held-out episodes**; exact normalization uses training episodes only. Fixed 128-example held-out evaluation uses identical noise/timesteps, center crops and dropout disabled, for both online and EMA weights. It measures denoising MSE, not action error or simulator success.

This smaller-batch, single-seed diagnostic is not directly comparable at equal steps to the stopped batch-8,960 run. Initial full-GPU baseline/identity losses matched exactly; maximum prediction difference was below `3e-6` from the wider matrix accumulation. The harness, CPU tests and GPU smoke passed before launch (harness commit `386fc5f`).

```bash
source /tmp/dev/env.sh
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTORCH_ALLOC_CONF=expandable_segments:True WANDB_BASE_URL=https://api.wandb.ai \
  taskset -c 90-119 .venv/bin/python scripts/b1k/compare_language_init.py \
  --dataset-path /tmp/dev/datasets/2026-challenge-demos \
  --output-dir /tmp/dev/audits/act-dp-identity-init-20260917/dp-1000 \
  --device cuda:0 --max-steps 1000 --batch-size 512 --num-workers 8 \
  --eval-samples 128 --eval-batch-size 16 --eval-every 250 \
  --wandb-mode online --wandb-entity kmy17518 \
  --wandb-project b1k-challenge-2026-diffusion-policy --wandb-group dp-init-20260917
```

Use a new output directory for another experiment; the harness rejects overwrite/resume. W&B runs: baseline [`e15394481fef`](https://wandb.ai/kmy17518/b1k-challenge-2026-diffusion-policy/runs/e15394481fef), random FiLM [`03d3cdad2e2e`](https://wandb.ai/kmy17518/b1k-challenge-2026-diffusion-policy/runs/03d3cdad2e2e), identity FiLM [`da0d66e0cc52`](https://wandb.ai/kmy17518/b1k-challenge-2026-diffusion-policy/runs/da0d66e0cc52). These are finite diagnostics with no HF uploader or recurring monitoring.

**Stopped intentionally after about 31.5 minutes when the user capped the experiment at one hour.** The common comparison endpoint is **step 500**, fully recorded for all three arms including held-out evaluation; later partial training is excluded. No final DP checkpoint was saved because the original 1,000-step target was interrupted (exit130). The existing long-run checkpoints remain untouched.

Steps401–500 mean training denoising MSE: baseline **0.108824**, random FiLM **0.114095** (+4.84%), identity FiLM **0.113668** (+4.45%). Step500 held-out EMA denoising MSE: baseline **0.037039**, random FiLM **0.042136** (+13.76%), identity FiLM **0.042203** (+13.94%). Online held-out MSE: **0.035663 / 0.042982 / 0.041695**, respectively. Identity improves training MSE by only0.37% relative to random FiLM and is effectively tied on held-out EMA MSE (+0.16%); it does not remove the conditioned gap in this short test. These are single-seed results on128examples from20heldout episodes, not simulator success or proof about the stopped original large-batch run. Analysis and curves: `/tmp/dev/audits/act-dp-identity-init-20260917/comparison-results.{json,md}` and `comparison-curves.png`. All comparison training and monitoring are stopped.

## Original unconditioned run

## Configuration

- Dataset: `/tmp/dev/datasets/2026-challenge-demos`, **turning_on_radio only**, all 200 episodes; exact full-task normalization, no episode limit.
- Target: **300,000 optimizer steps**.
- Actual policy: `DiffusionTransformerHybridImagePolicy`, **12 transformer layers, embedding width 512, 8 attention heads**.
- Three independent robomimic ResNet18/spatial-softmax camera encoders, GroupNorm, 96x96 input and 86x86 training crop.
- Observation history 2, prediction horizon 16, executed action horizon 8; categorical task input and original 23-D R1Pro actions.
- DDPM, 100 train/inference diffusion timesteps, epsilon target, global observation conditioning, causal attention, attention dropout 0.3.
- AdamW (FP32 parameters, gradients, optimizer state and EMA), learning rate `1e-4`, weight decay `1e-6`, betas 0.9/0.999. EMA weights are used for serving. No gradient accumulation. Since 2026-09-17 the recipe runs the forward pass under bf16 autocast with TF32 matmuls and `torch.compile` (see `b1k.md`, "Throughput"); the first 3,802 steps of the existing run were FP32 eager.
- Physical batch **8,960**, one loader task per optimizer batch (no slicing needed with the frame cache).
- GPU 3 (`GPU-2f892c97-af70-7c60-d2fa-456c65bd90ce`), CPU affinity **90-119**, 8 one-thread workers reading the pixel-exact frame cache `/tmp/dev/datasets/2026-challenge-demos-frame-cache-96` (built and verified by the launch script), two main torch threads, prefetch factor 1, episode cache 200 per worker.

The 130-CPU container budget leaves 30 cores for each of the two other training runs, 30 for ACT, 30 for DP, and approximately ten for upload/system work. Process affinity bounds these jobs; it is not an exclusive OS CPU reservation.

## Batch qualification

Physical FP32 probes passed 512, 8192, 8704, 8960 and 9088; 9120, 9152 and 9216 ran out of memory. A fresh-data pipeline run at 9088 ran out of memory on step 5 despite passing a repeated-input probe, so it was rejected. **8960 passed a 16-step fresh-data run** including online W&B, exact stats, three-checkpoint retention, and eval exports. In FP32 eager mode compute was approximately 2.2 seconds/step and native-video loading under the 30-core limit brought end-to-end steps to roughly 10-11 seconds (the run logged 10.3 s/step over its first 3,802 steps).

These measurements are in `/tmp/dev/audits/act-dp-radio-300k-20260916/`. Batch selection used practical aligned sizes near the OOM boundary, not every individual integer.

**Throughput work of 2026-09-17** (`/tmp/dev/audits/dp-speed-20260917/`): the pixel-exact frame cache removes the loader bottleneck entirely (data wait ~1 ms), and bf16 autocast + TF32 + `torch.compile` + multi-tensor EMA + plain-matmul attention bring the same recipe to **0.60 s/step at batch 8,960** (157 GiB peak instead of 268) and **0.083 s/step at batch 1,024** (`BATCH_SIZE=1024`; the script selects CUDA graphs below 4,096 samples). A 60-step run at 8,960 reproduced the original run's losses (steps 1-5 within ~1e-3; mean over steps 6-60 0.3752 vs 0.3753). The launch script below now builds/verifies the cache first and passes the new flags; relaunching resumes `latest.pt` with them.

## Comparison run with the throughput work (2026-09-17)

A fresh 300k-step run with the same architecture, batch (8,960), seed and data, launched with the optimized trainer to compare against the original run above:

```bash
tmux -L b1k-act-dp new-session -d -s dp-radio-fast-train \
  'RUN_TAG=fast-20260917 GPU_UUID=GPU-2246c972-301d-6778-7a38-bdf1d0e05687 CORES=40-69 \
   bash /tmp/dev/baselines/diffusion_policy/scripts/b1k/run_radio_300k.sh'
```

- Directory `outputs/turning-on-radio-transformer12x512-bs8960-300k-fast-20260917/`, log `/tmp/dev/logs/dp-radio-300k-bs8960-fast-20260917.log`, W&B run `dpradio16-bs8960-fast-20260917` (same project), GPU 1, trainer commit `100b106`.
- First 133 steps: 0.604 s/step versus 10.38 s/step originally; mean loss over steps 11-50 and 51-133 identical to four decimals (0.3641, 0.1800) since both runs see the same batches.
- No uploader was started for this run (the existing one is bound to the original run's Hub repo); its checkpoints accumulate locally under `export_queue/`.

## Detached training and uploader

Dedicated tmux server socket name: **`b1k-act-dp`**.

```bash
source /tmp/dev/env.sh
tmux -L b1k-act-dp list-sessions
tmux -L b1k-act-dp attach -t dp-radio-train
tmux -L b1k-act-dp attach -t dp-radio-upload
```

Reproducible launch recipes (trainer first, separate sessions):

```bash
tmux -L b1k-act-dp new-session -d -s dp-radio-train \
  'bash /tmp/dev/baselines/diffusion_policy/scripts/b1k/run_radio_300k.sh'
tmux -L b1k-act-dp new-session -d -s dp-radio-upload \
  'bash /tmp/dev/baselines/diffusion_policy/scripts/b1k/upload_radio_300k.sh'
```

The training script resumes its latest full checkpoint if present. Inspect and archive any prior `.exit` status before a deliberate restart. Existing-run locks and an occupied-GPU check reject accidental overlapping launches. Tmux survives Grok disconnect/session exit, not machine/container loss.

## Artifacts and retention

- Local directory: `outputs/turning-on-radio-transformer12x512-bs8960-300k-20260916/`.
- Full checkpoint at step 1 and every **2,500 steps**; retain newest **three full local checkpoints**.
- Eval-only EMA export every **10,000 steps**; retain all 30 scheduled eval snapshots remotely.
- Private Hugging Face: https://huggingface.co/kmy17518/b1k-dp-transformer12x512-turning-on-radio-20260916
- W&B: https://wandb.ai/kmy17518/b1k-challenge-2026-diffusion-policy/runs/dpradio16
- Uploader journal/staging: `/tmp/dev/hf-staging/dp-radio-300k-20260916/`.
- Logs: `/tmp/dev/logs/dp-radio-300k-20260916.log` and `/tmp/dev/logs/dp-radio-upload-20260916.log`.
- Corresponding `.exit` files record process termination; no such file means inspect tmux/status to determine liveness.

One uploader owns the dedicated Hub repo and both eval/full publication. The current tree has exactly one `resume/step-XXXXXXXX.pt`. Replacing it also removes the old path in the same commit, then permanently deletes only its recorded stale LFS object after checking ownership, refs, complete live-tree hashes, and exclusion of all current/eval objects. The local durable journal makes ambiguous network responses recoverable. Live scratch testing verified storage-object removal and retained eval integrity; actual step-1 upload was checked by SHA-256.

**Preserve the staging journal and keep this checkpoint repo single-writer/main-only.** Foreign files, branches, tags or PR refs stop collection safely. Network errors retry; safety failures exit 2 and need inspection. The uploader exits once the final full checkpoint and all 30 evals have been verified.

```bash
source /tmp/dev/env.sh
CUDA_VISIBLE_DEVICES='' /tmp/dev/baselines/act/.venv/bin/python /tmp/dev/scripts/act-dp-radio-status.py
```

The run is in progress; 300k steps are not yet complete. Simulator evaluation requires a separate RTX-capable host.
