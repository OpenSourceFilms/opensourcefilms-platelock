# Tested environments and evidence boundaries

**Updated:** 27 October 2026. This is a register of actual public evidence,
not a claim that a new GPU, a full production shot or Nuke has been qualified.
"Not recorded" means no measurement is available: it is **not** zero.

| Evidence | Execution environment | GPU / peak VRAM | Input dimensions | Frames | Wall time | Verifiable outcome |
|---|---|---|---|---|---|---|
| [Published synthetic registration CI](https://github.com/OpenSourceFilms/opensourcefilms-platelock/actions/runs/40333413408), 27 Oct | GitHub Actions ubuntu-latest, Python 3.11, NumPy 2.2.6, OpenCV-headless 4.11.0.86 | No GPU requested; VRAM not measured | Invented 236 x 132 BGR plate; generated three-panel PNG | 1 synthetic source frame and its translated copy | Per-method runtime not measured | CPU backward-map registration of known (+16,-10) translation; 0 interior MAE after mapping; five unit tests and image export passed. |
| [Historical fresh-venv integration](../RUNPOD.md#validation-this-pass-2026-08-19), 19 Aug | Fresh Python environment with cu128 PyTorch installed; actual methods executed on CPU fallback | **No NVIDIA GPU attached** in that validation; VRAM not applicable | Exact source dimensions not documented | 5 synthetic input pairs per tested method | Not recorded | ECC, SEA-RAFT and CoTracker mesh completed, produced expected metadata and previews. This did not validate CUDA kernels on PlateLock hardware or result quality. |
| [Synthetic smoke-wrapper CI](https://github.com/OpenSourceFilms/opensourcefilms-platelock/actions/workflows/smoke-wrapper.yml) | GitHub Actions, Python 3.11 and Bash command stubs | No GPU requested; VRAM not measured | Synthetic stubs, no image inference | Expected manifest count 5 | Not measured | Wrapper fails closed on missing metadata/nonzero exit and preserves distinct run output directories; **not** tracking-model acceptance. |

### What still has to be measured

The following evidence is **not yet present in this public repository**:

- A fresh installation and real kernel invocation on a named GPU and driver
  version, including CUDA/PyTorch build and library revision pins.
- Per-method actual peak VRAM, elapsed time and verified output hashes for
  a recorded image resolution, frame count and temporal window.
- A representative tracking/registration sequence with visual evaluation of
  edge alignment, motion continuity, occlusions and failure frames.
- Model/checkpoint provenance and rights clearance for any distributed
  weights or third-party media. CoTracker3 carries **CC-BY-NC-4.0** terms.
- Downstream Nuke STMap read/warp acceptance against the same synthetic or
  explicitly cleared frame source.

### Minimum credible GPU acceptance record

For any later GPU row record at least: pod GPU model, VRAM capacity and peak
allocation, driver and CUDA versions, Python/PyTorch/OpenCV versions, method,
model/checkpoint identifiers, source dimensions, frame count, elapsed runtime,
source and output SHA-256 digests, and the exact test command/exit status.
Report quality observations separately from process success. Never borrow
performance numbers from a different model or an unrelated ComfyUI pipeline.

This inventory deliberately does not count an installed CUDA wheel or
an unrelated GPU test as proof that PlateLock inference ran on GPU.
