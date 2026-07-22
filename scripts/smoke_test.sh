#!/bin/bash
# smoke_test.sh - Quick sanity check for each installed method.
# Generates 5 synthetic frames and runs each method. Outputs to output/smoke_tests/

BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$BENCH/env/platewarp"
export BENCH
OUT="$BENCH/output/smoke_tests"
SCRIPTS="$BENCH/scripts"

if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
    echo "Using venv: $VENV"
else
    echo "[WARN] Venv not found at $VENV; using system Python"
fi

echo
echo "== Python / torch / GPU =="
python3 -c "
import torch
print('torch:', torch.__version__, '| CUDA:', torch.version.cuda)
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(), '| cap:', torch.cuda.get_device_capability())
    t = torch.ones(1).cuda()
    print('CUDA tensor OK')
else:
    print('WARNING: no CUDA')
"

echo
echo "== Creating synthetic test frames =="
python3 << 'PYEOF'
import numpy as np, cv2, os
from pathlib import Path

base = Path(os.environ["BENCH"])
orig_dir = base / "output/smoke_tests/input/original"
clean_dir = base / "output/smoke_tests/input/clean"
orig_dir.mkdir(parents=True, exist_ok=True)
clean_dir.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
for i in range(5):
    # original: textured frame with slight motion pattern
    frame = np.random.randint(40, 200, (360, 640, 3), dtype=np.uint8)
    cv2.putText(frame, f"ORIGINAL {i:02d}", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
    cv2.imwrite(str(orig_dir / f"frame{i:04d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    # clean: same with a small global shift (simulates slight misalignment)
    M = np.float32([[1,0,3],[0,1,2]])
    shifted = cv2.warpAffine(frame, M, (640, 360))
    cv2.imwrite(str(clean_dir / f"frame{i:04d}.png"), cv2.cvtColor(shifted, cv2.COLOR_RGB2BGR))

print(f"Created 5 frame pairs in {orig_dir.parent}")
PYEOF

echo
echo "== Smoke test: ECC =="
python3 "$SCRIPTS/run_method.py" \
    --method ecc \
    --original "$OUT/input/original" \
    --clean "$OUT/input/clean" \
    --out "$OUT/ecc" \
    --config "$BENCH/configs/debug_5frames.yaml" && echo "ECC: PASS" || echo "ECC: FAIL"

echo
echo "== Smoke test: SEA-RAFT (PTLFlow) =="
python3 "$SCRIPTS/run_method.py" \
    --method searaft \
    --original "$OUT/input/original" \
    --clean "$OUT/input/clean" \
    --out "$OUT/searaft" \
    --config "$BENCH/configs/debug_5frames.yaml" && echo "SEARAFT: PASS" || echo "SEARAFT: FAIL"

echo
echo "== Smoke test: CoTracker mesh =="
python3 "$SCRIPTS/run_method.py" \
    --method cotracker_mesh \
    --original "$OUT/input/original" \
    --clean "$OUT/input/clean" \
    --out "$OUT/cotracker_mesh" \
    --config "$BENCH/configs/debug_5frames.yaml" && echo "COTRACKER_MESH: PASS" || echo "COTRACKER_MESH: FAIL"

echo
echo "== Results in $OUT =="
ls -la "$OUT"
