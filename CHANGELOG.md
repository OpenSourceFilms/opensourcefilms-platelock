# Changelog

## [Unreleased] - 2026-08-19

Deployment-presence pass: a tested, reproducible Runpod setup path. No functional or quality
changes to the pipeline itself.

### Added

- `RUNPOD.md`: recommended GPU class, dependency reproducibility notes, persistent-volume
  guidance, and a "Run on Runpod" section in `README.md`.
- `requirements.lock.txt`: pins the benchmark harness's direct Python dependencies to versions
  verified by a real fresh-environment install and smoke test (`ecc`, `searaft`, `cotracker_mesh`
  all ran end-to-end and produced expected output artifacts).

### Changed

- `scripts/setup_env.sh`: pins `torch`/`torchvision` to the versions already verified in
  production (`2.8.0`/`0.23.0`, cu128), drops the unused `torchaudio` install, installs from
  `requirements.lock.txt` instead of an inline package list, and pins the CoTracker3 clone to a
  specific verified commit (`82e02e8`) instead of tracking `main`.
- `scripts/methods/cotracker_mesh.py`: pins the `torch.hub.load` CoTracker3 reference to the same
  commit, so an upstream push can't silently change tracking output.
- `README.md`: removed the "no pinned dependency lock file" line from Known limitations (now
  addressed).

## [0.1.0] - 2026-07-22

Initial release. Work in progress, published as it stands.

### Added

- `platelock/`: the production tracking pipeline (tracking, dense reconstruction, temporal
  microlock smoothing, occlusion hole-fill, 5K export), config-driven via YAML recipes in
  `platelock/configs/`.
- `scripts/`: a benchmark harness comparing tracking methods (CoTracker3, RoMa, SEA-RAFT, FlowSeek,
  TAPNext, classical baselines), with shared metrics and a robust correspondence fitter.
- `configs/`: top-level benchmark harness configs.
- `LICENSE` (MIT) and `THIRD_PARTY_LICENSES.md`.

### Curated for public release

- Excluded a set of roughly 50 early Cython prototype files (`_mk*.pyx`, `_hf_*.py`) that predate
  the current `platelock/` package and were never wired into it: no code anywhere imports them, and
  no compiled build of them exists. The shipped, current pipeline is pure Python.
- Excluded a one-off launch script tied to a specific, still-in-progress production run, and the
  project's internal decision log (operational notes, not general documentation).
- Parametrized 3 hardcoded SAM3 checkpoint paths (`platelock/segmenters/sam3_segmenter.py` and 2
  YAML configs) via a `PLATELOCK_SAM3_CHECKPOINT` environment variable.
- Fixed hardcoded absolute deployment paths in `scripts/setup_env.sh`, `scripts/smoke_test.sh`, and
  `reinstall_deps.sh` to resolve relative to the script's own location instead.
- Generalized a real production shot's name (referenced throughout as an example) and internal
  decision-attribution comments to generic placeholders, without changing any technical content.
