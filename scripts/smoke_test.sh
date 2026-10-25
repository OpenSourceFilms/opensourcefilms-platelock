#!/usr/bin/env bash
# Exit nonzero if any method fails or provides incomplete native metadata.
set -uo pipefail
FAIL=0
# smoke_test.sh - Quick sanity check for each installed method.
# Generates 5 synthetic frames and runs each method. Outputs to output/smoke_tests/

BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$BENCH/env/platewarp"
export BENCH
mkdir -p "$BENCH/output/smoke_tests"
OUT="$(mktemp -d "$BENCH/output/smoke_tests/run_XXXXXXXX")" || exit 2
export OUT
printf "SMOKE OUTPUT: %s\n" "$OUT"
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
" || { echo "ENV: FAIL (torch/CUDA probe failed)"; exit 1; }

echo
echo "== Creating synthetic test frames =="
if ! python3 << 'PYEOF'
import numpy as np, cv2, os
from pathlib import Path

base = Path(os.environ["OUT"])
orig_dir = base / "input/original"
clean_dir = base / "input/clean"
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
then
    echo "INPUT: FAIL (synthetic generation failed)"
    exit 1
fi

verify_method() {
    python3 - "$OUT/$1/method_result.json" "$1" <<'PYCHECK'
import json,sys
from pathlib import Path
path=Path(sys.argv[1])
method=sys.argv[2]
if not path.is_file() or path.is_symlink():
    raise SystemExit("missing native method_result.json")
try:
    result=json.loads(path.read_text())
except (OSError,ValueError):
    raise SystemExit("invalid native result JSON")
if (result.get("method") != method or result.get("frame_count") != 5
    or not isinstance(result.get("metrics_summary"),dict)
    or not all(k in result["metrics_summary"] for k in
               ("mean_psnr","mean_ssim","mean_mae"))):
    raise SystemExit("incomplete or mismatched native result")
print("Native five-frame result verified:",method)
PYCHECK
}

echo
echo "== Smoke test: ECC =="
python3 "$SCRIPTS/run_method.py" \
    --method ecc \
    --original "$OUT/input/original" \
    --clean "$OUT/input/clean" \
    --out "$OUT/ecc" \
    --config "$BENCH/configs/debug_5frames.yaml" && verify_method ecc && echo "ECC: PASS" || { echo "ECC: FAIL"; FAIL=1; }

echo
echo "== Smoke test: SEA-RAFT (PTLFlow) =="
python3 "$SCRIPTS/run_method.py" \
    --method searaft \
    --original "$OUT/input/original" \
    --clean "$OUT/input/clean" \
    --out "$OUT/searaft" \
    --config "$BENCH/configs/debug_5frames.yaml" && verify_method searaft && echo "SEARAFT: PASS" || { echo "SEARAFT: FAIL"; FAIL=1; }

echo
echo "== Smoke test: CoTracker mesh =="
python3 "$SCRIPTS/run_method.py" \
    --method cotracker_mesh \
    --original "$OUT/input/original" \
    --clean "$OUT/input/clean" \
    --out "$OUT/cotracker_mesh" \
    --config "$BENCH/configs/debug_5frames.yaml" && verify_method cotracker_mesh && echo "COTRACKER_MESH: PASS" || { echo "COTRACKER_MESH: FAIL"; FAIL=1; }

echo
echo "== Results in $OUT =="
ls -la "$OUT" || FAIL=1
if [ "$FAIL" -eq 0 ]; then
    echo "smoke_test: PASS (all three methods produced native five-frame metadata)"
else
    echo "smoke_test: FAIL (method error or missing/mismatched native metadata)"
fi
exit "$FAIL"
