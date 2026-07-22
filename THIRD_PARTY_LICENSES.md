# Third-party licenses

Platelock's own code (everything in this repository) is MIT-licensed, see `LICENSE`. This
repository does not vendor any third-party model source code or weights; it depends on the
libraries and models below at runtime, installed separately (see `scripts/setup_env.sh` and
`reinstall_deps.sh`). Their licenses are their own and are not overridden by Platelock's MIT
license.

**Do not deploy this pipeline commercially without checking every dependency you actually use.**
Which of these load depends on which tracking method and config you run.

## Verified

| Component | License | Commercial use | Verified |
|---|---|---|---|
| [CoTracker3](https://github.com/facebookresearch/co-tracker) | CC-BY-NC 4.0 | **No**, non-commercial only | Documented in this repo's own README prior to this release; standard, well-known CoTracker3 license |
| RoMa | Apache-2.0/MIT (RoMa/FlowSeek family) | Yes | Prior audit (2026-07-22 session, this org) |
| SEA-RAFT | BSD-3-Clause | Yes | Prior audit (2026-07-22 session, this org) |
| FlowSeek | Apache-2.0/MIT | Yes | Prior audit (2026-07-22 session, this org) |
| [Meta SAM3](https://github.com/facebookresearch/sam3) | Custom "SAM License" (source-available; requires redistributing the license text and a publication acknowledgment; no revenue/MAU cap) | Yes, commercially usable | Prior audit (2026-07-22 session, this org) |

## Not yet verified

Used by this pipeline but not individually re-audited for this release. **Do not treat these as
cleared for commercial use** without checking the actual upstream license yourself:

- PTLFlow (the wrapper package used to run SEA-RAFT/FlowSeek here); the underlying models above are
  verified, the wrapper package's own license was not separately checked
- TAPNext / BootsTAPNext (`tapnext_mesh` method)
- MAST3R (`mast3r` method, future wave, not yet wired in)
- Meta SAM2 (used as a fallback segmenter alongside SAM3; commonly Apache-2.0, not independently
  re-verified this session)
- DINOv2 (`dinov2_vitl14_pretrain.pth`, a RoMa dependency; commonly Apache-2.0, not independently
  re-verified this session)

## Standard permissive libraries (not individually audited)

General-purpose libraries used throughout: `torch`, `torchvision`, `torchaudio`, `numpy`, `scipy`,
`opencv-python`, `scikit-image`, `imageio`, `pandas`, `tqdm`, `matplotlib`, `kornia`, `einops`,
`timm`, `pyyaml`, `huggingface_hub`, `safetensors`. These are well-established BSD/MIT/Apache-2.0
projects with no known non-commercial restriction. Check the specific version you install if in
doubt.

## How to update this file

If you add a new tracking method or model dependency, add a row above with its license, a
commercial-use verdict, and how you checked it. Prefer the actual upstream `LICENSE` file over a
summary elsewhere.
