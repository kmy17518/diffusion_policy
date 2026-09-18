# Native Diffusion Policy for BEHAVIOR-1K

This adapter trains directly on **LeRobot v3 packed parquet + video**. It does not convert the dataset to Zarr/HDF5 or replace upstream models. The only derived artifact is the optional, byte-for-byte verified resized-frame cache described under [Throughput](#throughput-loader-and-blackwell-gpu); the native video reader remains the default. Language conditioning is optional ([CLIP + FiLM](#optional-clip-language--film)); the default uses no language model. `--variant` selects the actual upstream diffusion policy class. Five classes are executable; the sixth, video, has genuinely missing upstream encoder source (see the coverage matrix below). The original Hydra entrypoints/configurations remain available.

The backward-compatible default is `unet_image`: upstream `DiffusionUnetImagePolicy`, `MultiImageObsEncoder`, `ConditionalUnet1D`, and DDPM with a shared ResNet18/GroupNorm encoder, no pretrained download, and RGB limits normalization. Old B1K v1 checkpoints without a variant field retain exactly that selection.

## Environment

Use a project-local environment in the checkout you are working in: the main clone or a git worktree of it (for example `/tmp/dev/baselines/diffusion_policy_lang_goal`). Each checkout gets its own `.venv` (gitignored) and its own `outputs/`; `scripts/b1k/run_radio_300k.sh` likewise runs from whichever checkout contains it. The commands on this page run from that checkout, so set `DP_DIR` to it once per shell. On the verified ARM/GB300 host, Python 3.11 avoids the unavailable Python 3.10 `numcodecs` wheel/header combination:

```bash
source /tmp/dev/env.sh
export DP_DIR=/tmp/dev/baselines/diffusion_policy   # this checkout; e.g. /tmp/dev/baselines/diffusion_policy_lang_goal for that worktree
cd "$DP_DIR"
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/bin/python -r requirements-b1k.txt
uv pip install --python .venv/bin/python -e .
```

On this ARM host, `uv pip check` reports a platform warning for `nvidia-cusparselt-cu13==0.8.0`, whose wheel declares the nonstandard `manylinux2014_sbsa` tag. The installed library is an aarch64 binary and loads successfully; real GPU training and serving passed. This remains a dependency-check warning.

Choose a hardware-appropriate PyTorch index elsewhere. The two hybrid classes require **robomimic 0.2.0**, matching the upstream environment (0.3.0 moves `CropRandomizer` and is incompatible). It is pinned in `requirements-b1k.txt`; lowdim and plain image models import independently of robomimic. There is no MuJoCo, pytorch3d, LeRobot/Hugging Face dataset runtime, or full simulator requirement for the supported B1K paths. The legacy HDF5 image loader imports its optional rotation transformer only when `abs_action=True`; its default raw-action behavior is unchanged.

## Training

```bash
source /tmp/dev/env.sh
cd "$DP_DIR"
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/b1k/train_b1k.py \
  --dataset-path /tmp/dev/datasets/2026-challenge-demos \
  --task-names turning_on_radio \
  --output-dir "$DP_DIR/outputs/radio" \
  --max-steps 100000 --batch-size 64 --num-workers 4 --device cuda
```

`--dataset-root` aliases `--dataset-path`. Omit `--task-names` to use all locally present episodes/tasks; multiple names are space-separated. Unknown names and requested tasks without episodes fail explicitly. Noncontiguous/nonzero episode IDs and partial downloads are supported. No missing data is downloaded. Missing selected camera files fail before training. The output must be outside the dataset tree, and an existing nonempty output requires `--resume`.

Defaults: trajectory horizon 16, observation history 2, executed prediction steps 8, images 96×96, all three cameras, U-Net widths 256/512/1024, diffusion embedding 256, cosine beta schedule, 100 training/inference noise steps, epsilon prediction, AdamW learning rate 1e-4. `--scheduler ddim --num-inference-steps 10` chooses DDIM. `--cameras head left_wrist` avoids loading the omitted camera. `--image-size`, `--horizon`, `--n-obs-steps`, `--n-action-steps`, `--down-dims`, and `--diffusion-step-embed-dim` are configurable. Horizon must be divisible by the U-Net downsampling factor. `--max-episodes` explicitly limits data for debugging; it is recorded in the checkpoint and must not eliminate a requested task.

### Optional CLIP language + FiLM

Add `--language-conditioning clip_film --prompt-source task_name` to an image recipe, or use `--prompt-source task_description`. Defaults are **`none`** and **`task_name`**, so old model/checkpoint tensor keys and categorical state conditioning are unchanged. The existing 25-D state plus task one-hot remains present in language mode too.

- `task_name` encodes the **exact raw metadata name**, including underscores and case; it does not replace underscores with spaces. `task_description` joins selected IDs/names to `meta/tasks.jsonl` records with `task_index`, `task_name`, and description in `task`. Duplicate/conflicting task mappings, malformed metadata, and missing/blank selected prompts fail before training. Different tasks may legitimately share description text.
- The initial run lazily loads frozen, evaluation-mode `CLIPTextModelWithProjection` from **`openai/clip-vit-large-patch14`**, pinned to **`32bd64288804d66eefd0ccbe215aa642df71cc41`**. It stores the **unnormalized 768-D `text_embeds`**, not normalized cosine features. CLIP is used once per selected task at setup on CPU, is never an optimizer parameter, and is not loaded by workers, resume, or serving. `transformers>=4.46,<5` is a versioned optional-runtime dependency in `requirements-b1k.txt`.
- CLIP has a 77-token context. Short prompts use standard tokenizer/model behavior. Longer prompts are tokenized **without truncation**, split into consecutive chunks of at most **75 content tokens**, wrapped individually with BOS/EOS, padded to 77 with attention masks, encoded, and reduced by an arithmetic mean of projected chunk embeddings. No token is silently dropped; the mean is not L2-normalized. The checkpoint records this scheme as `clip_77_content_chunks_mean_v1`.
- Each ResNet18 residual block (all eight, for every camera) is followed by a separate `Linear(768, 2*C)` producing **beta then gamma**, and `ReLU((1 + gamma) * x + beta)`. Language is an explicit forward argument, never a mutable global or hook. `lang_emb` is also concatenated alongside visual features and state as identity-normalized low-dimensional input, matching the RoboCasa conditioning path. Hybrid encoders retain independent cameras, ResNet18, 32-keypoint spatial-softmax, the 64-D projection/ReLU, and crop semantics; tests copy upstream weights and check zero-FiLM feature/gradient parity.
- Supported: `transformer_hybrid_image` global conditioning (including action-only), `unet_hybrid_image`, and `unet_image` (global/inpainting). The 12-layer/512-wide transformer uses the same controls as before. Lowdim/video reject `clip_film`. Transformer-hybrid inpainting rejects it because the upstream branch detaches the encoded trajectory. Frozen vision encoders reject it because FiLM must train. These restrictions apply only to language mode.
- GroupNorm language encoders automatically use `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` for each residual-block/FiLM pair during gradient-enabled training. Recomputation retains the **same physical batch and numerics as the surrounding run** (FP32 by default; the opt-in `--autocast bf16` / `--compile` / `--sdpa-backend` settings under [Throughput](#throughput-loader-and-blackwell-gpu) apply to the FiLM encoders too, and the recomputed blocks compile inside the encoder graph), with no gradient accumulation. BatchNorm paths and inference do not recompute blocks, avoiding double running-statistic updates. No-language behavior is untouched.
- Measured with the optimized recipe (frame cache, bf16, TF32, `torch.compile`, math attention; `/tmp/dev/audits/dp-lang-goal-20260917/` and `b1k_runs.md`): `clip_film` with `task_name` costs **0.094 vs 0.081 s/step at batch 1,024** (+17%) and **0.67 vs 0.60 s/step at batch 8,960** (+12%), with ~1 ms data wait either way; peak memory at 8,960 is **129 GiB vs 157 GiB** unconditioned because the eight FiLM blocks per camera are recomputed rather than stored. Same-seed losses track the unconditioned run 3–7% higher over the first 150 steps (mean over steps 6–60 at 8,960: 0.3908 vs 0.3752), consistent with the FiLM initialization comparison recorded in `b1k_runs.md`; convergence and task success are not established by these runs.
- `--no-film-recompute` (recipe `FILM_RECOMPUTE=off`) stores the FiLM blocks' activations instead of recomputing them. It is exact — in eager mode the loss and all 470 gradient tensors are bitwise equal either way, and the trainer test checks bitwise model/EMA equality after two steps — and at batch 8,960 it returns the step time to the unconditioned **0.59 s** for **172 GiB** peak (batch 1,024: 0.090 vs 0.094 s, 21 vs 16 GiB). Under `torch.compile` the two settings trace to different graphs, so their losses agree only to ~1e-5 relative, the same order as the recipe's own run-to-run nondeterminism (`cudnn.benchmark`). `--film-init identity` (recipe `FILM_INIT=identity`) zeroes the FiLM projections so each conditioned block starts as the identity (the harness's "identity FiLM" arm); fresh runs only.

Both full and eval checkpoints contain a top-level **`language`** dictionary: format, model, exact revision, prompt source, tokenization version, ordered task IDs/names, exact prompts, and finite float32 `[tasks, 768]` embeddings. Missing/invalid/mismatched language caches fail closed. Full-checkpoint resume uses this cache without reading description metadata or downloading CLIP; it still needs the selected demonstration data for training. Serving needs **neither CLIP nor the dataset** and maps known task IDs to saved embeddings; task changes reset history/action queues as before. The handshake exposes `language_conditioning` and `prompt_source`.

Programmatic setup: construct `ModelConfig(language_conditioning='clip_film', prompt_source=...)`, construct the dataset with `config.dataset_kwargs()`, call `dataset.prepare_language()` (or pass a saved `checkpoint['language']`), then use `build_policy(config, dataset.task_map)` and the dataset normalizer as usual. `dataset.language` supplies checkpoint export metadata. `load_policy` attaches the validated cache for `B1KPolicySession`; no external tokenizer/model is needed.

Offline tests: `source /tmp/dev/env.sh; CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest tests/test_b1k_language.py -q`. Fake text encoders cover exact prompt selection, malformed metadata, token boundaries/masks, both prompt modes through training/full/eval checkpoint/exact resume/dataset-free WebSocket serving, FiLM sensitivity and gradients, upstream hybrid parity, recomputation gradient parity, and no-language compatibility. The combined language/B1K/upload/replay-buffer/CV2/timestamp regression passed **243 tests**, with one existing opt-in private-HF upload test skipped. A separate cached real-CLIP CPU `prepare_language()` smoke passed both radio prompt sources. These checks do not claim simulator success or full-batch GPU qualification.

### Variant and recipe controls

| `--variant` | Actual upstream class | `--conditioning` | Action-only prediction |
| --- | --- | --- | --- |
| `unet_image` (default) | `DiffusionUnetImagePolicy` | `global`, `inpainting` | No |
| `unet_hybrid_image` | `DiffusionUnetHybridImagePolicy` | `global`, `inpainting` | No |
| `transformer_hybrid_image` | `DiffusionTransformerHybridImagePolicy` | `global` (observation cross-attention), `inpainting` | `--pred-action-steps-only`, global only |
| `unet_lowdim` | `DiffusionUnetLowdimPolicy` | `global`, `local`, `inpainting` | `--pred-action-steps-only`, global only |
| `transformer_lowdim` | `DiffusionTransformerLowdimPolicy` | `global` (observation cross-attention), `inpainting` | `--pred-action-steps-only`, global only |
| `unet_video` | `DiffusionUnetVideoPolicy` | Not executable: upstream source absent | Not executable |

Both `--scheduler ddpm` and `--scheduler ddim` work with each source-complete class. `--prediction-type epsilon|sample` selects the upstream training target. All meaningful controls are saved in the checkpoint and restored rather than inferred from a filename:

- U-Net: `--kernel-size`, `--n-groups`, `--cond-predict-scale` / `--no-cond-predict-scale`, diffusion embedding and widths. Only U-Nets impose downsampling divisibility; action-only U-Net checks the action length, not the unused full horizon.
- Transformer: `--n-layer`, `--n-head`, `--n-emb`, `--n-cond-layers`, `--p-drop-emb`, `--p-drop-attn`, `--causal-attn` / `--no-causal-attn`, `--time-as-cond` / `--no-time-as-cond`. A positive conditioning-layer count selects the upstream transformer conditioning encoder instead of its MLP. `--no-time-as-cond` requires inpainting and selects the upstream encoder-only architecture.
- Hybrid image: `--crop-shape H W`, `--obs-encoder-group-norm` and `--eval-fixed-crop` (both boolean options also accept `--no-...`). These instantiate robomimic's actual ResNet18/spatial-softmax encoders, not `MultiImageObsEncoder` substitutes.
- Plain image: `--share-rgb-model` / `--no-share-rgb-model`, `--resize-shape H W`, `--crop-shape H W`, `--random-crop` / `--no-random-crop`, `--obs-encoder-group-norm` / `--no-obs-encoder-group-norm`, `--imagenet-norm`. The independent-camera/GN/random-crop/ImageNet recipe is distinct from the backward-compatible shared B1K default.
- Pretrained/frozen plain image recipe: `--encoder-weights IMAGENET1K_V1 --freeze-encoder --no-obs-encoder-group-norm --imagenet-norm --no-random-crop --resize-shape 256 256 --crop-shape 224 224`. Initial weights are optional, and new pretrained training may download the official torchvision ResNet18 weights. Frozen encoders stay in evaluation mode, including BatchNorm buffers. Resume/serve builds with **no external weight initialization**, then restores checkpoint weights, so no download is needed. The real-data matrix checks a smaller 40→32 crop, not a default-resolution claim. R3M is only a commented upstream alternative, not a shipped discrete recipe, and is not supported/tested here.

`--imagenet-norm` records identity RGB linear normalization in the checkpoint: encoder ImageNet normalization expects [0,1], not the old adapter's [-1,1]. Old configurations keep [-1,1]. Lowdim data uses `batch['obs']` as a tensor and an `obs` normalizer instead of an image dictionary. During inference it passes the upstream `{'obs': tensor}` contract.

B1K parquet row t pairs observation t with action t. The adapter therefore explicitly sets **`oa_step_convention=True` for `DiffusionUnetLowdimPolicy`**, selecting `n_obs_steps-1` for both inference and action-only training. Other supported classes already use this convention. The original class's default remains unchanged.

Upstream behavior is preserved: transformer-hybrid inpainting detaches the entire encoded trajectory, so that branch does **not** train its image encoder end to end. The upstream U-Net local-conditioning second residual branch is intentionally unused for published-checkpoint compatibility and is not changed here.

The only variant-discovered upstream fixes are slicing causal transformer masks to the actual short action-only trajectory (full-horizon behavior unchanged), and assigning deterministic `CenterCrop` to the crop transform rather than a subsequently overwritten normalizer variable. Both have direct regression tests. Separately, the B1K trainer copies BatchNorm running statistics/counters into its EMA model after the upstream parameter-only EMA update. This supports the optional trainable-BatchNorm image recipe without changing the original EMA implementation or GroupNorm defaults; pretrained frozen BatchNorm remains unchanged. No unrelated BET, IBC, or robomimic imitation policies are substituted or included in DP scope; robomimic here is only an encoder dependency.

The B1K trainer runs in one process on one GPU (or CPU); it does not implement distributed training with `torchrun`. `--num-workers` controls CPU data-loading workers. The trainer records training loss, not held-out validation loss or simulator success.

`--max-steps` is the **total optimizer-step target**, including resumed steps:

```bash
source /tmp/dev/env.sh
cd "$DP_DIR"
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/b1k/train_b1k.py \
  --dataset-root /tmp/dev/datasets/2026-challenge-demos \
  --output-dir "$DP_DIR/outputs/radio" \
  --max-steps 110000 --device cuda --num-workers 4 \
  --resume "$DP_DIR/outputs/radio/latest.pt"
```

Resume restores architecture, normalization, task selection, model/EMA parameters and EMA counter, AdamW state, seed, batch size, learning rate/weight decay, and Python/NumPy/Torch/CUDA RNG states. Explicit model flags are ignored in favor of the checkpoint, except conflicting explicit `--language-conditioning` or `--prompt-source` values fail rather than silently changing language semantics. The frame-uniform replacement sampler is independently seeded per optimizer step, so prefetching does not change resumed batch indices. Dataset metadata, selected IDs, file sizes and nanosecond mtimes are fingerprinted to detect selection/mutation mismatches (not a full 3 TB content hash). GPU kernels may still be nondeterministic. Checkpoints use atomic `step-XXXXXXXX.pt` writes and a `latest.pt` symlink; existing step files are never overwritten. `config.json` and `train.jsonl` are run-local.

### Long runs, checkpoints, and tracking

By default the trainer computes in FP32 with no architecture changes or gradient accumulation; `--autocast bf16`, `--matmul-precision high` and `--compile` are the opt-in numeric/execution settings measured under [Throughput](#throughput-loader-and-blackwell-gpu) (parameters, gradients, optimizer state and EMA stay FP32 in every mode). A 12-layer image transformer is selected explicitly with `--variant transformer_hybrid_image --n-layer 12 --n-emb 512 --n-head 8`. Set the physical `--batch-size` from a separate single-GPU memory **and full input-pipeline** probe, not from a denoiser-only estimate. For a 300,000-step radio run, use `--task-names turning_on_radio --max-steps 300000`; the trainer does not launch evaluations or uploaders itself.

- `--save-every 2500 --save-total-limit 3` keeps the three newest local resumable `step-XXXXXXXX.pt` files. The default retention limit is **0 (keep all)**, preserving old behavior. `latest.pt` points to the newest completed full checkpoint. `--save-first-step` additionally saves/publishes step 1; it defaults off and does not create an early evaluation export. The final optimizer step always saves a full checkpoint.
- `--export-every 10000` (default; 0 disables) writes `export_queue/eval/step-XXXXXXXX.pt` exactly on that cadence. These contain only format/type, model configuration, task map, normalizer, EMA state, step, and the language cache when enabled. They contain **no training model, optimizer, training metadata, or RNG**, load directly through `load_policy`/`serve_b1k.py`, and are explicitly rejected by `--resume`. Exporting is not simulator evaluation: a separate evaluator must consume the files.
- Every completed full checkpoint is hardlinked into `export_queue/full/step-XXXXXXXX.pt` **before local retention pruning**. The full queue keeps only its newest file. A separate uploader must first hardlink the discovered queue file into its own staging directory on the same filesystem, retry discovery on `FileNotFoundError`, and then hash/upload the staged inode. That hardlink remains valid when the trainer removes older queue/local names. Evaluation exports are never pruned by the trainer; only an uploader with a verified remote commit should acknowledge/remove them.
- Checkpoint files are flushed/fsynced and atomically renamed; queue and symlink directory updates are fsynced. An advisory exclusive nonblocking `run.lock` protects the entire trainer lifetime, including resume. The lock file is deliberately retained after exit; deleting an active lock file would defeat writer exclusion.
- `--wandb-mode disabled|offline|online` defaults to **disabled**, requiring no W&B import or network access. Enable online tracking with `--wandb-mode online --wandb-project PROJECT --wandb-entity ENTITY --wandb-name NAME`; optional `--wandb-id ID` controls the initial run ID. Set `WANDB_API_KEY` via the sourced environment. Online mode fails clearly on missing credentials, initialization/authentication failure, or offline fallback **before constructing the dataset/model**. W&B is pinned in `requirements-b1k.txt`; do not run `wandb login` on this host.
- The stable W&B ID/project/entity/name are stored in `wandb.json` before initialization and in every full checkpoint. Resume restores identity and rejects conflicting ID/project/entity overrides. Pass `--wandb-mode online` again when resuming online tracking; omission intentionally leaves tracking disabled. `config.json` records effective runtime arguments. `train.jsonl` and W&B share step, loss, gradient norm, learning rate, elapsed time, `data_wait_s`, `compute_s`, `step_s`, `checkpoint_s`, samples/second, and GPU allocated/reserved/peak allocated bytes. CPU runs report zero GPU bytes. GPU timing synchronizes completion; `step_s` excludes checkpoint/logging time, which is reported separately for checkpoint writes.

The default optimizer remains generic AdamW: learning rate `1e-4`, weight decay `1e-6`, betas `(0.9, 0.999)`. To use the upstream transformer parameter grouping explicitly, choose `--optimizer upstream --learning-rate 1e-4 --weight-decay 0.001 --obs-encoder-weight-decay 0.000001 --betas 0.9 0.95`. This calls the actual policy's `get_optimizer`: transformer decay/no-decay groups plus separate image-encoder decay. Lowdim transformers use their corresponding upstream optimizer; non-transformer policies reject this option. Optimizer kind, betas, and encoder decay are restored from full checkpoints; older checkpoints restore the original generic AdamW recipe.

The upstream learning-rate schedule, EMA exponent and (absence of) gradient clipping are separate flags with the native trainer's historical behavior as defaults: `--lr-scheduler constant|cosine` with `--lr-warmup-steps N` (upstream `get_scheduler`: linear warmup, then cosine decay to zero over `--lr-schedule-steps`, default `--max-steps`; stepped once per optimizer step; the schedule length is fixed at the first launch and restored on resume, and the scheduler state is checkpointed so resume is exact), `--ema-power` (upstream `EMAModel` exponent, default 2/3; the RoboCasa recipe uses 0.75) and `--grad-clip` (default 1.0; `0` trains unclipped like the upstream workspace while still rejecting non-finite gradients). `train.jsonl` logs the learning rate applied at each step. Checkpoints written before these flags resume with constant LR, power 2/3 and clipping at 1.0.

**RoboCasa365 Diffusion Policy preset.** `scripts/b1k/run_radio_300k.sh` with `PRESET=robocasa365` reproduces the [RoboCasa365 Diffusion Policy baseline](https://github.com/robocasa-benchmark/diffusion_policy) configuration (`train_diffusion_transformer_bs192`): 256 px images with 224 px crops, horizon 10 / 2 observation steps / 8 executed actions, 12×512×8 transformer with a 4-layer transformer condition encoder (`--n-cond-layers 4`), CLIP ViT-L/14 FiLM language conditioning on the task description with **no one-hot task id** (`--no-task-onehot`), upstream parameter groups (AdamW 1e-4, betas 0.9/0.95, weight decay 1e-3 transformer / 1e-6 encoder), cosine schedule with 1000 warmup steps sized for 500k steps (their `num_epochs × max_train_steps`; the released checkpoint is epoch 500 = 250k steps, i.e. mid-decay), EMA power 0.75, no gradient clipping, batch 192, 250k steps. It keeps this trainer's throughput settings (frame cache keyed by image size, bf16 autocast, TF32, `torch.compile`, math attention, DDPM 100/100). Differences that remain: their data are 300 tasks with per-episode instructions, ours the selected B1K tasks with one description each; their robomimic encoder and ours are equivalent in structure (FiLM after each ResNet18 block, spatial softmax, 64-D projection).

With the frame cache, `--num-workers 8` and an unset `--loader-batch-size` (one worker task per optimizer batch, no main-process concatenation) keep the data wait near zero at both 1,024 and 8,960 samples per step. Without it, `--loader-batch-size 128 --num-workers 24 --prefetch-factor 1 --cpu-threads 2 --worker-cpu-threads 1` is an example input-pipeline configuration to benchmark for very large physical batches, not a fixed machine recommendation. Unset `--loader-batch-size` preserves one worker task per optimizer batch. When set, the sampler splits the same seeded sample-index list into bounded decoding chunks, spreads them across workers, then concatenates ordered chunks into **one unchanged physical optimizer batch** (including a shorter final chunk). Pinned worker tensors remain pinned after concatenation; there is no gradient accumulation or changed update cadence. Worker prefetch scales with the decoding chunk size rather than the physical optimizer batch. The defaults are prefetch factor 1, main CPU threads 2, and worker CPU threads 1; Torch, BLAS, OpenCV, and Arrow CPU/I/O pools are capped. PyAV decoders already use one thread. Per-batch dataset reads group episode/timestamp accesses for cache reuse and then restore the original sample order. For a small selected task, increasing `--episode-cache-size` to its episode count can avoid repeated parquet filtering; account for per-worker cache memory.

Offline regression command (no CUDA initialization or W&B/HF access):

```bash
source /tmp/dev/env.sh
cd "$DP_DIR"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 taskset -c 90-119 .venv/bin/python -m pytest \
  tests/test_b1k.py tests/test_replay_buffer.py::test tests/test_cv2_util.py \
  tests/test_timestamp_accumulator.py -q
```

The additional long-run tests cover retention 0/1/3, first-step saves, export cadence and dataset-free EMA serving, eval-resume rejection, uploader hardlinks surviving pruning, atomic-write interruption, concurrent-writer rejection, mocked W&B authentication/metrics/stable resume identity, CPU caps, bulk-read sample order, exact chunked/unchunked optimizer results, and upstream transformer optimizer grouping/resume.

### Throughput: loader and Blackwell GPU

Measured 2026-09-17 on one Blackwell GPU (sm_103, 276 GiB) with 30 Grace cores, for the 12-layer/512-wide `transformer_hybrid_image` radio recipe (three cameras, 96 px, 86 px crop, horizon 16). The original run logged **10.3 s/step at batch 8,960** (2.2 s compute, 8.1 s waiting for the 24-worker native video loader); the same recipe now takes **0.60 s/step**, and batch 1,024 goes from ~1 s/step (loader bound) to **0.083 s/step**. Evidence for every row lives under `/tmp/dev/audits/dp-speed-20260917/` (`gpu-b*.json`, `loader-*.json`, `runs/*/train.jsonl`).

| Stage (cumulative) | batch 1,024 | batch 8,960 | peak GPU memory (8,960) |
| --- | --- | --- | --- |
| Original: FP32 eager, native loader, 24 workers | ~1.0 s (loader bound) | 10.3 s (8.1 s data wait) | 268 GiB |
| Native loader fixes (LRU, seek, uint8 transport) | 0.63 s (loader bound) | ~5.6 s (loader bound) | 268 GiB |
| Frame cache: loader no longer limiting | 0.33 s (GPU bound) | 2.20 s (GPU bound) | 268 GiB |
| + TF32 matmuls, multi-tensor EMA update | 0.21 s | 1.59 s | 268 GiB |
| + `torch.compile` (single graphs for encoder and denoiser) | 0.14 s | 0.97 s | 257 GiB |
| + bf16 autocast, `cudnn.benchmark` | 0.14 s | **0.59 s** | 157 GiB |
| + CUDA graphs (`--compile reduce-overhead`) | 0.10 s | 0.75 s (slower; do not use) | 157 GiB |
| + plain matmul/softmax attention (`--sdpa-backend math`, default) | **0.083 s** | **0.60 s** (already chosen automatically) | 157 GiB |

Loader (pixel-exact, default on):

- `VideoReader` keeps up to `--video-max-open 32` decoders per worker instead of 3; each of this task's 10 packed videos was being reopened (~5 ms, index parse and probe) about twice per sample, a third of the per-sample cost.
- Seeks aim at `t + tolerance` instead of `t - tolerance` (only when the frame period exceeds `2 * tolerance`, always true at 30 fps): a backward keyframe seek can then never overshoot the wanted frame, and it no longer decodes a whole extra GOP whenever the wanted frame is itself a keyframe (6.5 -> 5.6 decoded frames per two-frame read; 0 mismatches over 1,200 compared reads).
- Images travel from the workers as uint8 and become float32 on the device through `images_to_float`, which divides by a 0-dim tensor so the result is bit-identical to the workers' `astype(np.float32) / 255.` (a Python-scalar divisor on CUDA multiplies by the reciprocal and is one ulp off for some of the 256 values). Loader IPC, pinning and host-to-device traffic shrink 4x. `B1KLeRobotDataset(image_dtype='float32')` remains the default for library users; the trainer requests `uint8`.
- The remaining floor is HEVC decoding itself, ~13 ms of one core per sample (720x720 head frames 1.2 ms each, 480x480 wrists 0.5 ms; GOP 8 means ~5.6 decodes per two frames): 24 workers reach ~1,600 samples/s, still below the GPU at either batch size.

Frame cache (pixel-exact, opt-in `--frame-cache DIR`):

```bash
source /tmp/dev/env.sh
cd "$DP_DIR"
CUDA_VISIBLE_DEVICES='' taskset -c 90-119 .venv/bin/python scripts/b1k/build_frame_cache.py \
  --dataset-path /tmp/dev/datasets/2026-challenge-demos --task-names turning_on_radio \
  --cache-dir /tmp/dev/datasets/2026-challenge-demos-frame-cache-96 --image-size 96 --workers 28 --verify 512
```

`build_frame_cache.py` decodes every selected packed video once, sequentially, through the same PyAV decoder and `resize_rgb` as the native reader, and stores `<cache>/videos/<camera>/chunk-XXX/file-YYY.frames.npy` (uint8 `[N, S, S, 3]`), `.pts.npy` (float64 presentation seconds) and a manifest with the source file size/mtime, image size and frame count. For the radio task (10 videos, 1.29 M frames) this took 200 s on 28 cores and 34 GB. `--verify N` then decodes N random training samples natively and compares every frame byte for byte (`verified 3072 cached frames from 512 samples ... identical`); the streams carry no B-frames, so decoding from the start and from a keyframe agree, and `pytest` repeats the comparison on synthetic videos. `FrameCacheReader.read` reproduces `VideoReader.read`'s timestamp rounding and 8 ms tolerance and raises the same error on unmatched timestamps. Readers and the dataset constructor validate manifests against the current source files and refuse stale, corrupt or mismatched-size entries, and the cache may not live inside the dataset tree. Training with the cache is bit-identical to native training (regression test on the synthetic dataset), the dataset fingerprint is unchanged, so resume works across cache and native runs, and 8 workers deliver 28k samples/s at batch 1,024 (0.28 ms per sample).

GPU (opt-in, chosen by the recipe script):

- `--matmul-precision high` uses TF32 tensor cores for the transformer matmuls; cuDNN convolutions default to TF32 in PyTorch already.
- `--multi-tensor-ema` (default on) updates the EMA with `_foreach_mul_`/`_foreach_add_`, the same two elementwise operations `EMAModel.step` issues per tensor, in a handful of launches instead of 850 (regression tests: bit-identical to upstream over several steps and in a trained checkpoint; BatchNorm and frozen parameters copied as upstream does). AdamW keeps PyTorch's default multi-tensor implementation; its fused kernel measured slower for these 425 tensors (5.1 vs 3.6 ms).
- `--compile default` compiles the observation encoder and denoiser in place (`module.forward = torch.compile(module.forward)`, so `state_dict` keys are untouched). Two upstream constructs prevented single graphs: `crop_randomizer.crop_image_from_indices` asserted its in-range crop offsets with `.item()` (a device sync each; now skipped only while compiling, offsets are drawn in range by construction) and drew the offsets with the host RNG plus a copy (now drawn on the images' device, same distribution); `nn.TransformerDecoder`/`Encoder` compared the registered causal mask against a generated one with `bool(tensor)` on every call (now told `is_causal` explicitly, which is exactly what that comparison concluded, with identical outputs; regression test). Attention over 16 (self) and 3 (cross) tokens is a poor fit for fused SDPA kernels: with the decoder layers still eager, cuDNN's 128x128-tile flash kernels cost 180 ms/step at batch 8,960. Once compiled, PyTorch's backend choice for that batch (71,680 batch x heads) falls back to plain matmul/softmax, which Inductor fuses (~5 ms/step equivalent), whereas at batch 1,024 it still picks cuDNN (~20 ms/step, a fifth of the step).
- `--sdpa-backend math` (default) therefore runs the training forward/backward under `torch.nn.attention.sdpa_kernel([MATH])`, so the decomposed attention is what gets traced and fused (0.10 -> 0.083 s/step at batch 1,024; no change at 8,960). The backend decision is baked into the traced graphs while neither the AOT-autograd nor the Inductor cache key includes the backend flags, so the trainer tags `torch.compiler.config.cache_key_tag` with the backend; without that, a cached graph from a run with the other setting is silently reused (this masked the effect during measurement until caches were disabled). `auto` restores PyTorch's own selection.
- `--autocast bf16` runs the forward pass in bf16 with FP32 GroupNorm/LayerNorm/softmax/loss (PyTorch's autocast policy); it halves activation memory (268 -> 157 GiB at batch 8,960).
- `--compile reduce-overhead` (CUDA graphs) removes launch overhead, which dominates at batch 1,024 (0.14 -> 0.10 s); at batch 8,960 it measured slower than `default` and is not used there. The trainer's step order (forward, backward, optimizer, release of the batch and loss) is what CUDA-graph trees require; 700 real-data steps ran cleanly. Should a run ever report "accessing tensor output of CUDAGraphs that has been overwritten", fall back to `--compile default`.
- `--cudnn-benchmark` (default on) lets cuDNN time convolution algorithms once; batch shapes are constant.
- Measured and rejected: `channels_last` (GroupNorm forces layout round trips), `max-autotune-no-cudagraphs` (slower than `default`), an eager or scatter-add max-pool backward (the remaining 15% hotspot; both slower inside the compiled graph).

Validation runs with the trainer on real data (`runs/` under the audit directory, same seed and therefore the same batches): at batch 1,024 for 400 steps, eager FP32 and `bf16 + compile reduce-overhead` start at losses 1.19705 vs 1.19732, agree on every 100-step mean loss within 0.0015, and their per-step difference (std 0.0073) is below the eager run's own step-to-step noise (0.0092), at 0.301 vs 0.100 s/step; the final configuration with math attention (300 steps, 0.083 s/step) agrees with the same reference within 0.0016 on every 100-step mean (per-step std 0.0077). At batch 8,960, `bf16 + compile default` reproduces the original run's first five losses to ~1e-3 (1.1988/1.2759/1.0615/0.9384/0.9092 vs 1.2005/1.2781/1.0593/0.9369/0.9073 logged by the original run) and its mean loss over steps 6-60 (0.3752 vs 0.3753), at 0.597 s/step with 1 ms data wait. Compiled dropout and random crops use Inductor's RNG stream rather than eager PyTorch's (`TORCHINDUCTOR_FALLBACK_RANDOM=1` restores eager streams at a 22% cost); noise and timestep sampling stay in eager code and remain seed-identical, as does everything else about resume.

### Dataset scaling and alignment

- Only compact episode metadata and needed camera columns are loaded at startup; dataset-wide advertised totals do not control local length.
- Sequence indexing uses O(episodes) cumulative counts, not an O(frames) index/permutation. Batches use O(batch size) sampling even for 210 million frames.
- Workers lazily read only state/action/timestamp/ID columns. Parquet row-group and episode caches are bounded by `--parquet-cache-mb` (default 256 MB per worker) and `--episode-cache-size` (8). A single uncached row group must still be decoded; a packed file with one large row group sets the transient minimum memory/I/O cost. Workers never hold the full dataset in RAM.
- Statistics scan each selected packed file once, streaming batches and filtering selected episodes. No video is decoded. Exact statistics for the full dataset require a real low-dimensional scan; there is no approximate or dataset-wide-statistics substitution.
- Global conditioning decodes only the first `n_obs_steps` images. Inpainting training requires full-horizon images and state; lowdim local/inpainting training similarly receives a full-horizon observation tensor. Lowdim policies read no video or camera metadata/files at all. PyAV seeks to preceding keyframes and validates presentation timestamps within 8 ms. Every camera uses its **own** `videos/<camera>/from_timestamp`, chunk and file; offsets are added in float64. Requested resized RGB frames and open decoders have bounded per-worker caches. No depth streams are read.
- Sequence bounds match upstream `SequenceSampler`, including clipping padding to `horizon-1`, short-episode exclusion when insufficient padding, edge replication, and no crossing episodes. Default padding is `n_obs_steps-1` before and `n_action_steps-1` after. Upstream prediction selects actions beginning at `n_obs_steps-1`, the current observation timestep.

### Robot and normalization contract

Input proprio is 61-D; the 25-D R1Pro state is exactly `[0:3, 53:57, 3:10, 24:26, 28:35, 49:51]`. Actions remain the original **23 joint/base commands**: velocity/absolute semantics are preserved. There is **no extra delta transform and no 6D rotation conversion**.

Categorical task conditioning is optional: with `--task-onehot`, one-hot channels ordered by sorted dataset task ID are appended to the 25-D state and remain literal 0/1 under normalization. The trainer CLI defaults to `--no-task-onehot` (state stays 25-D), so with several selected tasks the policy must get the task from `--language-conditioning clip_film` — the trainer rejects a multi-task configuration with no task signal. `ModelConfig.task_onehot` defaults to `True` so checkpoints written before the field existed keep their one-hot channels; the choice is recorded in the checkpoint and honored by the dataset, normalizer, policy input width and the server. Task maps are checkpoint-local and unseen IDs are rejected in both modes (the server still needs `task_id`, also to select the language embedding).

State and action use upstream `LinearNormalizer.fit(mode='limits')` semantics: per-dimension min/max to [-1,1], `range_eps=1e-4`, near-constant dimensions use unit scale and offset `-min`, and reported standard deviation uses sample variance. Streaming accumulation uses float64 and stores float32. RGB is converted from uint8 RGB/RGBA, alpha dropped, resized with centered zero padding using OpenCV bilinear interpolation, then CHW float32 /255; the upstream image normalizer maps [0,1] to [-1,1]. Training and serving share preprocessing.

The linear normalizer does **not clip** values outside the fitted range. DDPM/DDIM's upstream `clip_sample=True` clips the denoiser's predicted clean normalized sample; `--no-clip-sample` disables that scheduler setting. No post-unnormalization actuator clipping is added. Constant action dimensions consequently retain upstream unit-scale behavior rather than being forcibly projected to their demonstration value. The existing HDF5 loader still uses its original raw-action identity normalization when `abs_action=False`; B1K joint commands deliberately use the Zarr-style limits normalizer instead of pretending they are already normalized.

## Serving

```bash
source /tmp/dev/env.sh
cd "$DP_DIR"
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/b1k/serve_b1k.py \
  --model-path "$DP_DIR/outputs/radio/latest.pt" \
  --host 0.0.0.0 --port 8000 --device cuda --action-horizon 8
```

`--model-path` can also be a run directory containing `latest.pt`. Serving loads **EMA weights** and needs neither the dataset nor external normalization/config files. Checkpoints contain primitive values/tensors and are loaded with `torch.load(weights_only=True)`; still use checkpoints from trusted sources and avoid exposing an unauthenticated server to untrusted networks.

`--action-horizon` is the **replanning interval**, between 1 and the trained `n_action_steps`, not the trajectory length. Each observation request returns only the next action. The server sends a MessagePack metadata handshake, then accepts MessagePack-numpy observations:

| Key | Shape/type |
| --- | --- |
| `robot_r1::proprio` | `(61,)` or `(B,61)`, finite numeric |
| `robot_r1::robot_r1:zed_link:Camera:0::rgb` | `(H,W,3/4)` or `(B,H,W,3/4)`, uint8 |
| `robot_r1::robot_r1:left_realsense_link:Camera:0::rgb` | same; only needed if trained with left wrist |
| `robot_r1::robot_r1:right_realsense_link:Camera:0::rgb` | same; only needed if trained with right wrist |
| `task_id` | integer scalar or B integers |

For either lowdim variant, **all RGB keys are optional and ignored**; only proprio and task selection are required. The handshake reports the actual upstream class name and checkpoint variant.

Reply: `{"action": ndarray(B,23,float32)}`. Wire ndarray/scalar encoding matches the GR00T/OpenPi byte-key `__ndarray__`/`__npgeneric__` protocol and never falls back to pickle. Handshake task-map keys are strings for clients using strict MessagePack map keys. Any message containing `reset` clears the connection state **with no reply**. HTTP `GET /healthz` returns 200 `OK`.

Histories and action queues are per connection and per batch slot. Every observation updates history, even while a cached action is consumed. Initial/reset histories repeat the first frame, matching training padding. A task change clears only that slot; a batch-size change clears all slots. An omitted task defaults only for a single-task checkpoint, or when `--task-name` selects a known checkpoint task. Supplied task IDs are always validated, even with a default. Invalid requests close their connection with a clear error. Shared model/scheduler calls are serialized, with inference off the async network loop so health checks remain responsive. There is no environment-slot identity beyond stable batch positions; send reset when reusing slots for new episodes.

## Tests and real-data smoke

```bash
source /tmp/dev/env.sh
.venv/bin/python -m pytest tests/test_b1k.py -q
```

The suite covers nonzero/noncontiguous IDs, partial roots and missing tasks/files, differing nonzero camera offsets, cached decoder reuse, short episodes/start/end padding versus upstream Zarr `ReplayBuffer`/`SequenceSampler`, actual HDF5 ingestion, constant limits and no normalizer clipping, uint8/RGBA padding, upstream DDPM/DDIM backward/inference, exact current-action alignment, bounded sampling/resume, dataset-free EMA serving, real websocket handshake/health/reset/batched-client isolation, and task-change validation. Existing simulator/hardware-dependent tests are not prerequisites.

### Complete upstream recipe inventory and mapping

The repository contains **six diffusion policy classes, 15 `train_diffusion*.yaml` recipes, and 27 task YAMLs**. The matrix runner writes their full configuration values to `outputs/variants/cpu-matrix/inventory.json`. These are upstream task/architecture recipes, not 15 different denoisers. B1K replaces the task I/O dimensions and reader, not the underlying model classes. None of the legacy simulator datasets/action conversions below are claimed to be B1K evaluation runs.

All filenames below are under `diffusion_policy/config/`. `T/To/Ta` is upstream horizon / observation steps / action steps; DDPM defaults to 100 inference steps. Evidence case prefixes refer to `outputs/variants/cpu-matrix/<case>/evidence.json`.

| Full recipe filename | Upstream task; distinguishing configuration | B1K branch evidence mapping |
| --- | --- | --- |
| `train_diffusion_unet_image_workspace.yaml` | `lift_image_abs`; 16/2/8; independent ResNet18, GN, ImageNet normalization, random 76px crop, widths 512/1024/2048 | `unet_image-independent-imagenet-ddpm` |
| `train_diffusion_unet_image_pretrained_workspace.yaml` | `lift_image_abs`; shared ImageNet ResNet18, BN, frozen, 256→224 center crop | `unet_image-pretrained-frozen-ddpm` |
| `train_diffusion_unet_real_image_workspace.yaml` | `real_pusht_image`; DDIM/100; independent encoders, 240×320→216×288 random crop | `unet_image-independent-imagenet-ddim` |
| `train_diffusion_unet_real_pretrained_workspace.yaml` | `real_pusht_image`; DDIM/100; shared frozen pretrained, 224×224 resize, no crop | `unet_image-pretrained-frozen-ddim` (crop exercised as an additional branch) |
| `train_diffusion_unet_hybrid_workspace.yaml` | `lift_image_abs`; robomimic spatial-softmax, GN/fixed eval crop; widths 512/1024/2048 | `unet_hybrid_image-global-ddpm` |
| `train_diffusion_unet_ddim_hybrid_workspace.yaml` | `lift_image_abs`; DDIM/8; widths 256/512/1024 | `unet_hybrid_image-global-ddim` |
| `train_diffusion_unet_real_hybrid_workspace.yaml` | `real_pusht_image`; DDIM/8; rectangular 216×288 crop | `unet_hybrid_image-global-ddim` |
| `train_diffusion_transformer_hybrid_workspace.yaml` | `lift_image_abs`; 10/2/8; 8 layers, width 256, causal observation conditioning | `transformer_hybrid_image-global-ddpm` |
| `train_diffusion_transformer_real_hybrid_workspace.yaml` | `real_pusht_image`; 16/2/8; DDIM/8, rectangular crop | `transformer_hybrid_image-global-ddim` |
| `train_diffusion_unet_lowdim_workspace.yaml` | `pusht_lowdim`; 16/2/8; global U-Net, widths 256/512/1024 | `unet_lowdim-global-ddpm` |
| `train_diffusion_unet_ddim_lowdim_workspace.yaml` | `pusht_lowdim`; DDIM/8; observation inpainting (global/local conditioning disabled) | `unet_lowdim-inpainting-ddim` |
| `train_diffusion_transformer_lowdim_workspace.yaml` | `blockpush_lowdim_seed`; 5/3/1; 8 layers, width 256, dropout .3 | `transformer_lowdim-global-ddpm` |
| `train_diffusion_transformer_lowdim_pusht_workspace.yaml` | `pusht_lowdim`; 16/2/8; width 256, dropout .01 | `transformer_lowdim-global-ddpm` |
| `train_diffusion_transformer_lowdim_kitchen_workspace.yaml` | `kitchen_lowdim_abs`; 16/4/8; width 768, dropout .1 | `transformer_lowdim-global-ddpm` |
| `train_diffusion_unet_video_workspace.yaml` | `lift_image_abs`; 16/4/8; VideoResNet50 + VideoCore + TemporalAggregator | `unet_video-global-ddpm` and `-ddim`: explicit missing-source construction evidence |

These mappings establish **material branch coverage with reduced CPU widths/resolutions**, not tests of every recipe's default width, horizon, camera geometry, dropout, optimizer schedule, or original task. Those numerical settings remain configurable. Transformer conditioning-encoder/encoder-only and action-only modes are additional tested constructor branches even where YAMLs select the defaults.

Task-specific equivalents are inventoried without silently dropping any: `blockpush_lowdim_seed{,_abs}`; `kitchen_lowdim{,_abs}`; `pusht_image`, `pusht_lowdim`, `real_pusht_image`; and, for each of `can`, `lift`, `square`, `tool_hang`, `transport`, `{task}_image{,_abs}` and `{task}_lowdim{,_abs}`. They choose observation/action layouts, datasets and runners, not extra diffusion policy classes. BET, IBC, and robomimic imitation training YAMLs/classes are explicitly **outside DP scope**.

Video is **blocked**, not passed or hidden as a skipped test. Importing its actual policy fails with `ModuleNotFoundError: diffusion_policy.model.obs_encoder`. The shipped recipe also names missing `model/obs_encoder/video_core.py` (`VideoCore`, `VideoResNet`) and `model/ibc/global_avgpool.py`; the policy needs missing `model/obs_encoder/temporal_aggregator.py`. Repository history and the official upstream tree at `5ba07ac` were checked independently (`/tmp/dev/audits/act-diffusion-upstream-tree.json`). No authentic implementations were found. The explicit selector reports these missing sources; no torchvision video encoder or invented temporal aggregator is presented as equivalent. Video action alignment/training correctness cannot be verified until those authentic sources are available.

### Repeatable all-variant real-data CPU matrix

```bash
source /tmp/dev/env.sh
cd "$DP_DIR"
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m diffusion_policy.b1k.variant_matrix \
  --dataset-path /tmp/dev/datasets/2026-challenge-demos \
  --task-names turning_on_radio --max-episodes 2 \
  --output-dir outputs/variants/cpu-matrix --device cpu
```

Use a **new output directory** for a rerun. `--cases <exact-case-id> ...` selects cases; `--list` prints all configurations without training. `--device cpu|cuda` is explicit. For this follow-up all GPUs were occupied by unrelated jobs, so **only CPU ran with `CUDA_VISIBLE_DEVICES=''`**. CUDA remains an available runner option only for a separately verified idle GPU; no CUDA variant coverage is claimed here.

The CUDA matrix fails closed unless exactly one physical index or GPU UUID is explicitly selected in `CUDA_VISIBLE_DEVICES`. Before training, resume, and moving the restored policy onto CUDA, `nvidia-smi` must report no compute processes, at most 256 MiB used, and zero utilization. Query failures/unknown values reject the launch. The gate canonicalizes the selection to the queried GPU UUID; separate case workers release their CUDA contexts before the next check. These gates have CPU-only mocked regression tests, not a claim of successful CUDA execution or an atomic GPU reservation.

Each executable case launches the actual native trainer for two steps, saves a checkpoint, launches a separate resume process to total step three, checks optimizer/EMA state and changed denoiser tensors, restores the checkpoint with downloads forbidden, asserts the actual policy/scheduler classes, then serves over real localhost WebSockets using real parquet/video observations. The current runner extracts observations and closes/deletes its dataset reader before restore; during restore and network serving an audit hook denies Python opens beneath the dataset root and patched native reader entrypoints reject dataset/parquet/video construction. Existing evidence predating this guard is distinguished from the guarded reruns below. It verifies health, class/variant handshake, finite float32 actions, cached-action replanning, two-client isolation, no-reply reset, reset-history padding, and batch-size changes plus batched replanning. Lowdim wire requests contain **no RGB**. Frozen cases check encoder parameters and buffers unchanged between steps two and three.

Evidence is per-case JSON plus `train.log`, `resume.log`, `run/config.json`, `run/train.jsonl`, `run/step-00000002.pt`, `run/step-00000003.pt`, and `run/latest.pt`. `summary.json` aggregates cases. This is a nonconverged functionality test, not task success.

#### Completed coverage (2026-09-15)

**43 executable CPU cases passed; 2 video cases explicitly blocked; zero hidden skips or failed executable cases.** The original 42-case executable matrix plus two video probes completed in 457 seconds in `outputs/variants/cpu-matrix/summary.json`. The additional trainable-BatchNorm case passed in `outputs/variants/cpu-batchnorm/summary.json`; a fresh full runner now includes it. All cases used two real `turning_on_radio` episodes (IDs 0 and 1, 4,294 sequences), batch size 2, horizon 4, observation/action steps 2, 4 training diffusion timesteps/2 inference steps, and CPU threads 2.

| Actual class / material branch | DDPM | DDIM | Exact evidence case IDs / directory |
| --- | --- | --- | --- |
| `DiffusionUnetImagePolicy`, global and inpainting | Passed | Passed | `unet_image-{global,inpainting}-{ddpm,ddim}` |
| `DiffusionUnetHybridImagePolicy`, global and inpainting | Passed | Passed | `unet_hybrid_image-{global,inpainting}-{ddpm,ddim}` |
| `DiffusionTransformerHybridImagePolicy`, observation-conditioned, inpainting, action-only | Passed | Passed | `transformer_hybrid_image-{global,inpainting,action-only}-{ddpm,ddim}` |
| `DiffusionUnetLowdimPolicy`, global, local, inpainting, action-only | Passed | Passed | `unet_lowdim-{global,local,inpainting,action-only}-{ddpm,ddim}` |
| `DiffusionTransformerLowdimPolicy`, observation-conditioned, inpainting, action-only | Passed | Passed | `transformer_lowdim-{global,inpainting,action-only}-{ddpm,ddim}` |
| Both transformer classes, encoder-only and learned conditioning encoder, noncausal attention | Passed | Passed | `{transformer_lowdim,transformer_hybrid_image}-{encoder-only,cond-encoder}-{ddpm,ddim}` |
| Independent image encoders, GN, random crop, ImageNet normalization | Passed | Passed | `unet_image-independent-imagenet-{ddpm,ddim}` |
| Shared ImageNet-pretrained frozen BN encoder, deterministic crop | Passed | Passed | `unet_image-pretrained-frozen-{ddpm,ddim}` |
| U-Net sample target, additive conditioning, kernel 3 | Passed | Passed | `unet_lowdim-sample-target-{ddpm,ddim}` |
| Trainable BN image encoder, EMA running-buffer synchronization | Passed | Not separately tested | `outputs/variants/cpu-batchnorm/unet_image-trainable-batchnorm-ddpm/evidence.json` |
| `DiffusionUnetVideoPolicy` actual import and construction | **Blocked: missing source** | **Blocked: missing source** | `unet_video-global-{ddpm,ddim}` |
| CUDA follow-up matrix / default-size recipe sweep | **Not run** | **Not run** | No idle GPU; reduced-width CPU evidence only |

Except the explicitly named extra BN directory, case IDs resolve to `outputs/variants/cpu-matrix/<case>/evidence.json`. There are **344 finite WebSocket action replies** across executable cases. Each JSON asserts actual class, checkpoints at 2/3, restored optimizer/EMA state, no restore download, and no initialized CUDA context. CPU U-Nets use widths **16/32**, timestep embedding **16**; transformers use **one layer, width 32, two heads**. Encoders remain real ResNet18/spatial-softmax models, with 32px input, hybrid 28px crop, and recipe resize/crop 40→32. Most image cases use head RGB; independent-encoder cases use head plus left wrist. These are not claims of default-width/default-resolution training.

`outputs/variants/pretrained-reference.json` additionally verifies all pretrained encoder weights/running statistics exactly equal official torchvision `IMAGENET1K_V1` values after step 3, with all BatchNorm counters still zero (120 encoder state tensors). The frozen encoder is not randomly initialized and mislabeled pretrained.

Final local regression command: `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest tests/test_b1k.py tests/test_replay_buffer.py::test -q --junitxml=outputs/variants/pytest-final.xml`. It passed **113 tests** (112 B1K plus one legacy replay-buffer test), preserving all original 45 B1K tests. Evidence: `outputs/variants/pytest-final.log` and `.xml`. It includes exact resumed-versus-uninterrupted state equality for six representative variants, tensor/full-horizon/no-video data contracts, old checkpoint defaults, offline restore, action-only training row alignment, and the two narrowly fixed upstream branches. Warnings are the existing NumPy padding warning, upstream torchvision deprecated pretrained argument, and transformer nested-tensor optimization warnings.

Safety-review reruns: `outputs/variants/cpu-dataset-denied/summary.json` records two additional passes (`unet_image-global-ddpm`, `transformer_lowdim-inpainting-ddim`) with `dataset_access_denied_during_restore_and_serve=true` and 16 additional finite network replies. These are reruns of existing variants, not extra distinct configurations. The final safety suite passed **126 tests** (125 B1K + one legacy replay-buffer test), including fail-closed CUDA gate tests with mocked `nvidia-smi` responses and dataset-denial guard tests; evidence is `outputs/variants/pytest-safety-final.log` and `.xml`. No CUDA launch was used to test the gate.

Independent main-agent evidence under `/tmp/dev/audits/act-diffusion-variants-20260915/` includes `upstream-lowdim.xml` (20 real-data upstream branch cases), `upstream-image.xml` (12 cases using all three real cameras at 84px), and `factory-equivalence.xml` (11 factory/data/normalization checks). Its `diffusion-independent-cli*` artifacts additionally verify native public train/resume/serve CLI behavior for an encoder-only transformer-lowdim inpainting DDIM checkpoint using the existing OpenPI codec, proprio-only requests, reset and batching. These do not replace the native trainer/resume/network evidence above.

### Original single-variant CUDA evidence (before this follow-up, 2026-09-15)

GPU 1 was checked idle with `nvidia-smi` before each CUDA launch. The smoke used the default architecture and image size, all three real cameras, and two selected episodes of `turning_on_radio` (4,282 training sequences):

```bash
source /tmp/dev/env.sh
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/b1k/train_b1k.py \
  --dataset-path /tmp/dev/datasets/2026-challenge-demos \
  --task-names turning_on_radio \
  --output-dir /tmp/dev/baselines/diffusion_policy/outputs/b1k-real-default \
  --max-steps 2 --batch-size 2 --num-workers 2 --device cuda --max-episodes 2 \
  > outputs/b1k-validation/train-real-default.log 2>&1

source /tmp/dev/env.sh
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/b1k/train_b1k.py \
  --dataset-root /tmp/dev/datasets/2026-challenge-demos \
  --output-dir /tmp/dev/baselines/diffusion_policy/outputs/b1k-real-default \
  --max-steps 3 --batch-size 2 --num-workers 2 --device cuda \
  --resume /tmp/dev/baselines/diffusion_policy/outputs/b1k-real-default/step-00000002.pt \
  > outputs/b1k-validation/resume-real-default.log 2>&1

source /tmp/dev/env.sh
.venv/bin/python -m pytest tests/test_b1k.py tests/test_replay_buffer.py::test -q \
  > outputs/b1k-validation/pytest-final.log 2>&1
```

Losses were 1.180549, 1.186948, and 1.151939 on steps 1, 2, and resumed 3. Checkpoints are `outputs/b1k-real-default/step-00000002.pt` and `step-00000003.pt` (`latest.pt`); the resumed checkpoint has EMA step 3 and 208 optimizer parameter states. The final regression run passed **46 tests**, with one upstream uint8 NaN-padding warning in the HDF5 comparison.

Real network smoke results are in `outputs/b1k-validation/network-real.log` (8 successful action requests). The reviewer-owned CUDA server used this command:

```bash
source /tmp/dev/env.sh
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/b1k/serve_b1k.py \
  --model-path /tmp/dev/baselines/diffusion_policy/outputs/b1k-real-default/latest.pt \
  --host 127.0.0.1 --port 18101 --device cuda --action-horizon 4
```

It served the resumed checkpoint for health, handshake, real parquet/video RGB/RGBA observations, replan, independent clients, no-reply reset, and batch-size checks. Independent wire verification exercised an 11-step action/replanning sequence plus reset, batching, and reconnect requests with the existing OpenPi codec, including unseen-task rejection. The reviewer stopped the server after verification, and GPU 1 was idle on the final check. Independent review also checked actual episodes 199, 8401, and 19692 at first/last frame, including six-hour packed-video offsets; all-task metadata (20,000 episodes, 100 tasks), partial roots, and full-task statistics.

This is a **nonconverged functionality smoke**, not a performance or task-success claim. Simulator evaluation is unavailable on this GB300 host because OmniGibson/BEHAVIOR requires RTX rendering support. Use a supported RTX evaluation machine connected to this policy server; the network checks do not establish simulator success. Stop the server after evaluation; no service is installed.
