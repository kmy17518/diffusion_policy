# Native Diffusion Policy for BEHAVIOR-1K

This adapter trains directly on **LeRobot v3 packed parquet + video**. It does not convert the dataset to Zarr/HDF5, write dataset caches, use language models, or replace upstream models. `--variant` selects the actual upstream diffusion policy class. Five classes are executable; the sixth, video, has genuinely missing upstream encoder source (see the coverage matrix below). The original Hydra entrypoints/configurations remain available.

The backward-compatible default is `unet_image`: upstream `DiffusionUnetImagePolicy`, `MultiImageObsEncoder`, `ConditionalUnet1D`, and DDPM with a shared ResNet18/GroupNorm encoder, no pretrained download, and RGB limits normalization. Old B1K v1 checkpoints without a variant field retain exactly that selection.

## Environment

Use a project-local environment. On the verified ARM/GB300 host, Python 3.11 avoids the unavailable Python 3.10 `numcodecs` wheel/header combination:

```bash
source /tmp/dev/env.sh
cd /tmp/dev/baselines/diffusion_policy
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
cd /tmp/dev/baselines/diffusion_policy
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/b1k/train_b1k.py \
  --dataset-path /tmp/dev/datasets/2026-challenge-demos \
  --task-names turning_on_radio \
  --output-dir /tmp/dev/baselines/diffusion_policy/outputs/radio \
  --max-steps 100000 --batch-size 64 --num-workers 4 --device cuda
```

`--dataset-root` aliases `--dataset-path`. Omit `--task-names` to use all locally present episodes/tasks; multiple names are space-separated. Unknown names and requested tasks without episodes fail explicitly. Noncontiguous/nonzero episode IDs and partial downloads are supported. No missing data is downloaded. Missing selected camera files fail before training. The output must be outside the dataset tree, and an existing nonempty output requires `--resume`.

Defaults: trajectory horizon 16, observation history 2, executed prediction steps 8, images 96×96, all three cameras, U-Net widths 256/512/1024, diffusion embedding 256, cosine beta schedule, 100 training/inference noise steps, epsilon prediction, AdamW learning rate 1e-4. `--scheduler ddim --num-inference-steps 10` chooses DDIM. `--cameras head left_wrist` avoids loading the omitted camera. `--image-size`, `--horizon`, `--n-obs-steps`, `--n-action-steps`, `--down-dims`, and `--diffusion-step-embed-dim` are configurable. Horizon must be divisible by the U-Net downsampling factor. `--max-episodes` explicitly limits data for debugging; it is recorded in the checkpoint and must not eliminate a requested task.

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
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/b1k/train_b1k.py \
  --dataset-root /tmp/dev/datasets/2026-challenge-demos \
  --output-dir /tmp/dev/baselines/diffusion_policy/outputs/radio \
  --max-steps 110000 --device cuda --num-workers 4 \
  --resume /tmp/dev/baselines/diffusion_policy/outputs/radio/latest.pt
```

Resume restores architecture, normalization, task selection, model/EMA parameters and EMA counter, AdamW state, seed, batch size, learning rate/weight decay, and Python/NumPy/Torch/CUDA RNG states. Explicit model flags are ignored in favor of the checkpoint. The frame-uniform replacement sampler is independently seeded per optimizer step, so prefetching does not change resumed batch indices. Dataset metadata, selected IDs, file sizes and nanosecond mtimes are fingerprinted to detect selection/mutation mismatches (not a full 3 TB content hash). GPU kernels may still be nondeterministic. Checkpoints use atomic `step-XXXXXXXX.pt` writes and a `latest.pt` symlink; existing step files are never overwritten. `config.json` and `train.jsonl` are run-local.

### Dataset scaling and alignment

- Only compact episode metadata and needed camera columns are loaded at startup; dataset-wide advertised totals do not control local length.
- Sequence indexing uses O(episodes) cumulative counts, not an O(frames) index/permutation. Batches use O(batch size) sampling even for 210 million frames.
- Workers lazily read only state/action/timestamp/ID columns. Parquet row-group and episode caches are bounded by `--parquet-cache-mb` (default 256 MB per worker) and `--episode-cache-size` (8). A single uncached row group must still be decoded; a packed file with one large row group sets the transient minimum memory/I/O cost. Workers never hold the full dataset in RAM.
- Statistics scan each selected packed file once, streaming batches and filtering selected episodes. No video is decoded. Exact statistics for the full dataset require a real low-dimensional scan; there is no approximate or dataset-wide-statistics substitution.
- Global conditioning decodes only the first `n_obs_steps` images. Inpainting training requires full-horizon images and state; lowdim local/inpainting training similarly receives a full-horizon observation tensor. Lowdim policies read no video or camera metadata/files at all. PyAV seeks to preceding keyframes and validates presentation timestamps within 8 ms. Every camera uses its **own** `videos/<camera>/from_timestamp`, chunk and file; offsets are added in float64. Requested resized RGB frames and open decoders have bounded per-worker caches. No depth streams are read.
- Sequence bounds match upstream `SequenceSampler`, including clipping padding to `horizon-1`, short-episode exclusion when insufficient padding, edge replication, and no crossing episodes. Default padding is `n_obs_steps-1` before and `n_action_steps-1` after. Upstream prediction selects actions beginning at `n_obs_steps-1`, the current observation timestep.

### Robot and normalization contract

Input proprio is 61-D; the 25-D R1Pro state is exactly `[0:3, 53:57, 3:10, 24:26, 28:35, 49:51]`. Actions remain the original **23 joint/base commands**: velocity/absolute semantics are preserved. There is **no extra delta transform and no 6D rotation conversion**.

Every policy has explicit categorical task conditioning: one-hot channels, ordered by sorted dataset task ID, are appended to the 25-D state. These remain literal 0/1 under normalization. Task maps are checkpoint-local; unseen IDs are rejected. This is not language conditioning.

State and action use upstream `LinearNormalizer.fit(mode='limits')` semantics: per-dimension min/max to [-1,1], `range_eps=1e-4`, near-constant dimensions use unit scale and offset `-min`, and reported standard deviation uses sample variance. Streaming accumulation uses float64 and stores float32. RGB is converted from uint8 RGB/RGBA, alpha dropped, resized with centered zero padding using OpenCV bilinear interpolation, then CHW float32 /255; the upstream image normalizer maps [0,1] to [-1,1]. Training and serving share preprocessing.

The linear normalizer does **not clip** values outside the fitted range. DDPM/DDIM's upstream `clip_sample=True` clips the denoiser's predicted clean normalized sample; `--no-clip-sample` disables that scheduler setting. No post-unnormalization actuator clipping is added. Constant action dimensions consequently retain upstream unit-scale behavior rather than being forcibly projected to their demonstration value. The existing HDF5 loader still uses its original raw-action identity normalization when `abs_action=False`; B1K joint commands deliberately use the Zarr-style limits normalizer instead of pretending they are already normalized.

## Serving

```bash
source /tmp/dev/env.sh
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/b1k/serve_b1k.py \
  --model-path /tmp/dev/baselines/diffusion_policy/outputs/radio/latest.pt \
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
cd /tmp/dev/baselines/diffusion_policy
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
