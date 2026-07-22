# Platelock

Locks an AI-generated clean plate onto original footage for VFX compositing. Produces STMaps for
Nuke's STMap node.

This is a work in progress. The core pipeline (tracking, reconstruction, microlock, hole-fill,
STMap export) is built and has shipped real deliverables, but it's still under active development.
Expect rough edges.

## Quick start

```bash
# 1. Set up environment (only needed once, fixes the Blackwell cu128 requirement)
bash scripts/setup_env.sh

# 2. Activate
source env/platewarp/bin/activate

# 3. Smoke test (5 synthetic frames)
bash scripts/smoke_test.sh

# 4. Run on real footage
python scripts/run_benchmark.py \
  --original /path/to/shot/original.mp4 \
  --clean /path/to/shot/clean.mp4 \
  --out output/some_shot \
  --methods cotracker_mesh dual_video ecc \
  --config configs/full_quality.yaml
```

For the production tracking pipeline (the more developed path, config-driven):

```bash
python -m platelock.run --config platelock/configs/tracking_multikey13.yaml --out output/run_example
```

## Method tiers (benchmark harness)

| Tier | Method | Type | Status | Notes |
|------|--------|------|--------|-------|
| **1** | `cotracker_mesh` | CoTracker3 tracks, RANSAC, piecewise affine | Ready | Best immediate production path |
| **1** | `tapnext_mesh` | TAPNext++/BootsTAPNext tracks, same fitter | Needs install | `pip install -e repos/tapnet[torch]` |
| **2** | `roma_v2` | RoMa v2 direct orig/clean dense matching | Needs install | `pip install romatch` |
| **3** | `mast3r` | 3D-grounded geometry matching | Future wave | For parallax-heavy shots |
| **4** | `flowseek` | FlowSeek (ICCV 2025) dense residual flow | Needs install | `pip install ptlflow` |
| **4** | `searaft` | SEA-RAFT dense residual flow | Needs install | `pip install ptlflow` |
| Baseline | `dual_video` | Phase correlation dual-video | Ready | Good classical baseline |
| Diagnostic | `ecc` | ECC translation only | Ready | Sanity check, not for final use |
| Diagnostic | `farneback` | LK+RANSAC per-frame | Ready | Classical smoke test |

**CoTracker3 is CC-BY-NC 4.0, non-commercial only.** RoMa, SEA-RAFT, FlowSeek, and PTLFlow are
Apache-2.0 or MIT. See `THIRD_PARTY_LICENSES.md`.

## Architecture

All track-based methods share the same downstream fitter (`scripts/utils/robust_fit.py`):

```
original sequence
  -> SAM2/SAM3 or diff mask (scripts/utils/mask_utils.py, platelock/masks.py)
  -> dense background point grid
  -> CoTracker3 or TAPNext++ tracks (within original and within clean)
  -> RANSAC outlier rejection
  -> model ladder: translation -> affine -> piecewise affine
  -> temporal smoothing
  -> STMap EXR/NPY -> Nuke STMap node
```

RoMa v2 uses the same fitter but with per-frame direct correspondences instead of tracks. The
`platelock/` package is the more developed, config-driven production pipeline: tracking, dense
reconstruction, temporal microlock smoothing, occlusion hole-fill, and 5K export, built around YAML
recipes in `platelock/configs/`.

## Metrics

PSNR/SSIM are logged but are **not the primary quality signal**. A method that improves PSNR but
adds 0.5px temporal jitter is a failure for compositing.

Compositing-appropriate metrics (`scripts/utils/compositing_metrics.py`):

| Metric | What it measures |
|--------|-------------------|
| `mean_chamfer_px` | Mean px distance between background Canny edges. Lower is better lock. |
| `mean_edge_recall_2px` | Fraction of original bg edges within 2px of warped edges. |
| `mean_jerk` | 2nd derivative of warp magnitude. Higher means swimming/jitter. |
| `mean_stmap_grad_u/v` | Spatial gradient of STMap channels. Higher means discontinuities. |

## GPU requirement

Development targeted an NVIDIA RTX PRO Blackwell card (sm_120). System torch builds tied to older
CUDA versions cannot run on sm_120; `scripts/setup_env.sh` creates a fresh venv with cu128 PyTorch.

## Nuke STMap convention

STMaps are exported as `.npy` float32 arrays (H x W x 2).

- **Backward map**: for each *output* pixel, stores the UV coordinate to sample from the source.
- **U** (channel 0) = source X / W, range [0, 1], left to right.
- **V** (channel 1) = 1 minus (source Y / H), a V-flip since Nuke's origin is bottom-left.
- Use R=U, G=V in Nuke's STMap node.

To use in Nuke: load the `.npy`, write to a 16/32-bit EXR (R=U, G=V), connect to an STMap node, use
as Source, apply to the clean plate.

## Adding new methods

All methods must implement:

```python
def process_sequence(self, originals: list, cleans: list, show_progress=True) -> dict:
    # Returns: warped_frames, flows, stmaps, debug_per_frame, method_name
```

Feed correspondences through `scripts/utils/robust_fit.fit_robust()` so all methods stay comparable.

## Configs

| Config | Use for |
|--------|---------|
| `configs/debug_5frames.yaml` | Fast 5-frame sanity check |
| `configs/fast.yaml` | 30 frames, half-res, quick comparison |
| `configs/default.yaml` | Standard nightly run: Tier 1 plus baselines |
| `configs/full_quality.yaml` | Final quality assessment, all tiers |
| `platelock/configs/*.yaml` | Production tracking pipeline recipes, see comments in each |

## SAM3 mask segmentation (optional)

`platelock/segmenters/sam3_segmenter.py` and the `platelock/configs/dense_crux*.yaml` masking
option use Meta's SAM3 model. Set `PLATELOCK_SAM3_CHECKPOINT` to your checkpoint's path, or edit the
`checkpoint:` field directly in the config you're using. This is opt-in and disabled by default in
most configs.

## What's not included

This is a code-only export from an internal working directory. Deliberately excluded:

- `output/`, `input/`, `env/`, `logs/`: real render outputs, real source footage, and the local
  Python virtual environment
- A set of early, superseded Cython prototype files that predate this package and were never wired
  into the shipped pipeline
- Internal session/decision-log notes

## Known limitations

Still under active development, published as it stands rather than as a finished deliverable.

- No pinned dependency lock file. `scripts/setup_env.sh` and `reinstall_deps.sh` install what's
  needed, but versions aren't pinned yet.
- The benchmark harness (`scripts/`) and the production pipeline (`platelock/`) are related but
  separate systems that grew somewhat independently; some duplication between them hasn't been
  reconciled.

## Acknowledgments

Development and GPU compute for this project are supported by Runpod, our compute sponsor. The
Blackwell-class cards this pipeline was built and tuned against came from Runpod pods.

## License

Open Source Films' own code in this repository (everything except the third-party dependencies
listed below) is released under the MIT License. See `LICENSE`.

This pipeline depends on third-party libraries and models at runtime, several of which are
non-commercial or otherwise restricted, most notably CoTracker3 (CC-BY-NC 4.0). See
`THIRD_PARTY_LICENSES.md` before any commercial deployment.
