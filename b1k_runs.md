# Transformer Diffusion Policy radio run — 2026-09-16

## Task-name CLIP/FiLM run

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

The 300,000-step trainer was launched on **2026-09-16 at 18:11 UTC** from commit `81839b5` using the new identities above. It is a fresh run, not the qualification checkpoint. Training is **in progress**, not complete; no simulator success rate is claimed. Local qualification and monitoring artifacts use `/tmp/dev/audits/act-dp-language-20260916/`. Tmux survives client disconnects, not machine/container termination.

## Original unconditioned run

## Configuration

- Dataset: `/tmp/dev/datasets/2026-challenge-demos`, **turning_on_radio only**, all 200 episodes; exact full-task normalization, no episode limit.
- Target: **300,000 optimizer steps**.
- Actual policy: `DiffusionTransformerHybridImagePolicy`, **12 transformer layers, embedding width 512, 8 attention heads**.
- Three independent robomimic ResNet18/spatial-softmax camera encoders, GroupNorm, 96x96 input and 86x86 training crop.
- Observation history 2, prediction horizon 16, executed action horizon 8; categorical task input and original 23-D R1Pro actions.
- DDPM, 100 train/inference diffusion timesteps, epsilon target, global observation conditioning, causal attention, attention dropout 0.3.
- FP32 AdamW, learning rate `1e-4`, weight decay `1e-6`, betas 0.9/0.999. EMA weights are used for serving. No mixed precision or gradient accumulation.
- Physical batch **8,960**. Loader slices of 128 are reassembled before the optimizer update without changing batch/sample order.
- GPU 3 (`GPU-2f892c97-af70-7c60-d2fa-456c65bd90ce`), CPU affinity **90-119**, 24 one-thread workers, two main torch threads, prefetch factor 1, episode cache 200 per worker.

The 130-CPU container budget leaves 30 cores for each of the two other training runs, 30 for ACT, 30 for DP, and approximately ten for upload/system work. Process affinity bounds these jobs; it is not an exclusive OS CPU reservation.

## Batch qualification

Physical FP32 probes passed 512, 8192, 8704, 8960 and 9088; 9120, 9152 and 9216 ran out of memory. A fresh-data pipeline run at 9088 ran out of memory on step 5 despite passing a repeated-input probe, so it was rejected. **8960 passed a 16-step fresh-data run** including online W&B, exact stats, three-checkpoint retention, and eval exports. Compute was approximately 2.2 seconds/step; native-video loading under the 30-core limit brought end-to-end steps to roughly 10-11 seconds. The requested largest stable physical batch is therefore CPU-input-bound; no throughput-optimal smaller batch was substituted.

These measurements are in `/tmp/dev/audits/act-dp-radio-300k-20260916/`. Batch selection used practical aligned sizes near the OOM boundary, not every individual integer. The actual detached run uses the same tested architecture, batch, data and optimizer configuration.

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
