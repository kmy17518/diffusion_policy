# Transformer Diffusion Policy radio run — 2026-09-16

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
