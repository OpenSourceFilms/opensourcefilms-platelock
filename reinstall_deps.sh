#!/usr/bin/env bash
# Run this after every pod restart to restore all ephemeral pip packages.
# All source code and weights are on persistent storage; this only
# reinstalls packages that live in /usr/local/lib (ephemeral).
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== reinstall_deps.sh: platelock/SAM3 pipeline ==="
echo "Restoring torch hub weights cache..."
mkdir -p /root/.cache/torch/hub/checkpoints
for f in roma_outdoor.pth dinov2_vitl14_pretrain.pth; do
    src="$ROOT/vendor/weights/torch_hub/$f"
    dst="/root/.cache/torch/hub/checkpoints/$f"
    if [ -f "$src" ] && [ ! -f "$dst" ]; then
        cp "$src" "$dst" && echo "  restored $f"
    elif [ -f "$dst" ]; then
        echo "  $f already present"
    else
        echo "  WARNING: $src not found, RoMa will re-download on first run"
    fi
done

echo "Installing pipeline Python deps (no-deps where safe)..."
pip install --quiet romatch
pip install --quiet scikit-image imageio imageio-ffmpeg pandas tqdm pyyaml lightglue timm ptlflow
pip install --quiet iopath ftfy regex pycocotools

echo "Installing SAM3 editable (source on persistent storage)..."
pip install --quiet --no-deps -e "$ROOT/vendor/sam3"

echo "Installing SAM2 (source on /workspace/repos if present, else from index)..."
SAM2_REPO="/workspace/repos/sam2"
if [ -d "$SAM2_REPO" ]; then
    pip install --quiet --no-deps -e "$SAM2_REPO"
else
    pip install --quiet --no-deps sam2
fi

echo "Installing tapnet + cotracker editable repos..."
for repo in /workspace/repos/tapnet /workspace/repos/co-tracker; do
    if [ -d "$repo" ]; then
        pip install --quiet --no-deps -e "$repo" && echo "  installed $repo"
    fi
done

echo ""
echo "=== Verifying key imports ==="
python3 - <<'PY'
import importlib, sys
checks = {
    "torch": "torch.cuda.is_available()",
    "romatch": None,
    "sam3.model_builder": "build_sam3_image_model",
    "cv2": None, "numpy": None, "scipy": None,
}
for mod, attr in checks.items():
    try:
        m = importlib.import_module(mod)
        extra = ""
        if attr and hasattr(m, attr.split("(")[0]):
            extra = f" [{attr}]"
        elif attr and "." not in attr:
            try: exec(f"from {mod} import {attr}"); extra = f" [import {attr} OK]"
            except: extra = f" [WARN: {attr} not found]"
        print(f"  OK  {mod}{extra}")
    except Exception as e:
        print(f"  FAIL {mod}: {e}", file=sys.stderr)
# cuda check
import torch
print(f"  cuda available: {torch.cuda.is_available()}, device_count: {torch.cuda.device_count()}")
PY

echo ""
echo "=== reinstall_deps.sh complete ==="
