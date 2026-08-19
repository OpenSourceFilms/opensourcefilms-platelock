# Running Platelock on Runpod

Platelock was developed and iterated on Runpod GPU pods. This document describes a tested path to
stand it up on a fresh pod, and states plainly what has and hasn't been verified.

## Recommended pod

- **Base image**: any Runpod PyTorch template is a reasonable starting point, but `setup_env.sh`
  builds its own isolated venv and installs its own pinned PyTorch, so a generic CUDA-capable
  Linux image works too. What matters is the driver, not a specific template.
- **GPU class**: development and tuning happened on an NVIDIA RTX PRO Blackwell card (sm_120).
  `setup_env.sh` installs PyTorch's cu128 build specifically because system/stock PyTorch builds
  tied to older CUDA releases cannot run Blackwell's sm_120 kernels. Any Runpod GPU that supports
  CUDA 12.8 should work; Blackwell-class cards (e.g. RTX PRO 6000 Blackwell) are the only ones
  this repo has actually been run and tuned on.
- **Driver**: needs to be recent enough for CUDA 12.8 (the 550-series or newer, per NVIDIA's
  published CUDA/driver compatibility table). Check with `nvidia-smi` on the pod before assuming;
  this wasn't independently re-measured this session.
- **VRAM**: no hard floor has been measured. The CoTracker3 tracking step deliberately runs at a
  modest 960x540 working resolution by default specifically to control VRAM use
  (`scripts/methods/cotracker_mesh.py`, `tracking_wh`), then scales tracks back up before the mesh
  warp. A mid-range Blackwell card is a safe default; if you're running the full-quality tier on
  long sequences and see OOMs, that's a tuning knob (`tracking_wh`, `chunk_size`), not evidence the
  approach needs more VRAM than expected.

## Setup

```bash
git clone https://github.com/OpenSourceFilms/opensourcefilms-platelock.git
cd opensourcefilms-platelock
bash scripts/setup_env.sh
source env/platewarp/bin/activate
```

`setup_env.sh` is idempotent (safe to rerun after a pod restart) and does, in order: GPU/driver
info dump, installs `ffmpeg` if missing, creates the venv, installs the pinned cu128 PyTorch build,
verifies a real CUDA kernel op against the GPU (not just `torch.cuda.is_available()` — that call is
known to under-report on some sm_120 builds; the script does an actual `.cuda()` tensor op instead),
installs the Python dependencies from `requirements.lock.txt`, and clones + installs CoTracker3
pinned to a specific verified commit (`82e02e8`) rather than tracking upstream `main`.

## Dependency reproducibility

- **PyTorch**: pinned to `torch==2.8.0` / `torchvision==0.23.0`, cu128 build. This exact pair was
  verified against a real CUDA kernel op on Blackwell hardware in production
  (`torch.randn(8,8,device="cuda")` then a matmul, not just an import check).
- **Benchmark harness packages**: `requirements.lock.txt` pins the direct dependencies
  (`ptlflow`, `opencv-python-headless`, `numpy`, `scipy`, `scikit-image`, `imageio`,
  `imageio-ffmpeg`, `tqdm`, `pandas`, `matplotlib`, `kornia`, `einops`, `timm`, `huggingface_hub`,
  `safetensors`, `PyYAML`) to versions verified by a real fresh-environment install and smoke test
  (see Validation below). This is not a full transitive freeze — only the packages that actually
  determine whether the harness runs.
- **CoTracker3**: cloned and installed from a pinned commit (`82e02e8`) instead of `main`, and
  loaded via `torch.hub.load("facebookresearch/co-tracker:82e02e8", ...)` in
  `scripts/methods/cotracker_mesh.py`, so an upstream push can't silently change tracking output.
  This is the same commit already pinned in the separate, more developed production pipeline.
- **SAM3** (optional, opt-in mask segmentation — most configs don't use it): not installed by
  `setup_env.sh`. If you need it: `git clone https://github.com/facebookresearch/sam3.git` and
  `pip install --no-deps -e sam3/`, then either download SAM3's checkpoint (~3.4GB) yourself and
  point `PLATELOCK_SAM3_CHECKPOINT` at it, or pass `masks.checkpoint` in your shot config. Not
  covered by this validation pass.

## Persistent volume guidance

If you're iterating across multiple pod sessions, put on a persistent volume:

- `env/` — the venv itself (rebuild is cheap via `setup_env.sh`, but skipping the rebuild saves
  time)
- `repos/co-tracker/` — avoids re-cloning
- `~/.cache/torch/hub/` — CoTracker3's downloaded weights, cached after first tracking run
- your own `input/`/`output/` shot data (never committed to this repo; see `.gitignore`)

Nothing in this repo writes outside its own directory tree except that hub cache.

## Smoke test

```bash
bash scripts/smoke_test.sh
```

Generates 5 synthetic frame pairs and runs `ecc`, `searaft`, and `cotracker_mesh` against them,
each producing metrics, comparison previews, and stills under `output/smoke_tests/`.

## Running a real workflow

Benchmark harness (compares tracking methods on a real shot):

```bash
python scripts/run_benchmark.py \
  --original /path/to/shot/original.mp4 \
  --clean /path/to/shot/clean.mp4 \
  --out output/some_shot \
  --methods cotracker_mesh dual_video ecc \
  --config configs/full_quality.yaml
```

Production tracking pipeline (config-driven, the more developed path):

```bash
python -m platelock.run --config platelock/configs/tracking_multikey13.yaml --out output/run_example
```

Neither command is bundled with sample footage in this repo (see README "What's not included").
Bring your own original/clean-plate video pair.

## Validation (this pass, 2026-08-19)

Run from a genuinely fresh venv following the steps above, in the environment available for this
task:

| Check | Result |
|---|---|
| Fresh venv creation | Pass |
| Pinned PyTorch cu128 install | Pass — imports correctly, reports the expected cu128 build |
| Benchmark harness dependency install (`requirements.lock.txt`) | Pass — all imports resolve |
| CoTracker3 clone + pin + install | Pass |
| `ecc` method, 5-frame smoke test | Pass — completed, produced metrics + previews |
| `searaft` method (PTLFlow), 5-frame smoke test | Pass — completed, produced metrics + previews |
| `cotracker_mesh` method, 5-frame smoke test | Pass — completed, produced metrics + previews |
| No silent dependency on this pod's own workspace layout | Confirmed — ran from a plain clone with no absolute paths outside the repo |
| Real GPU (CUDA) execution | **Not verified this pass** — see below |

**Honest gap**: the environment this validation ran in has no NVIDIA GPU attached (no
`nvidia-smi`, no CUDA device). The three methods above ran successfully on CPU fallback — which is
real, useful evidence the code doesn't hard-require CUDA to execute correctly — but this does not
substitute for GPU validation. The pinned `torch==2.8.0+cu128` build is carried over from the
separate production pipeline, where it was independently verified against a real CUDA kernel op on
actual Blackwell hardware (see that project's own environment notes) — not re-measured on GPU
hardware in this specific pass. Before trusting production-quality output (as opposed to "the code
runs"), rerun `scripts/smoke_test.sh` on an actual Runpod GPU pod and confirm the CUDA kernel check
in `setup_env.sh` Step 6 passes.

## Licensing note — read before any commercial deployment

`cotracker_mesh` — the tier-1, "best immediate production path" method — depends on **CoTracker3,
which is CC-BY-NC 4.0: non-commercial use only.** This is not a Platelock restriction; it comes
from the upstream model. Platelock's own code is MIT-licensed and imposes no such restriction.

If you need a path with no non-commercial encumbrance, the `dual_video` (classical phase
correlation) and `ecc`/`farneback` (diagnostic) methods are permissively licensed and ready to run,
though they are the classical baselines, not the strongest-quality tier. RoMa v2, SEA-RAFT, and
FlowSeek are Apache-2.0/MIT and commercially usable but are listed in the README as "needs install"
tiers, not validated in this pass. See `THIRD_PARTY_LICENSES.md` for the full picture, including
what's been independently license-verified versus what hasn't.

## Acknowledgments

GPU compute for developing and tuning this pipeline was provided by Runpod.
