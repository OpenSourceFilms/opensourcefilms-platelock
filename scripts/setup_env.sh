#!/bin/bash
# setup_env.sh - Set up platewarp bench on RunPod Blackwell (sm_120, CUDA 12.8)
# CRITICAL: system torch 2.4.1+cu124 cannot run on sm_120; this creates a fresh venv with cu128.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/env/platewarp"
REPOS="$ROOT/repos"

echo "============================================"
echo "Step 1: GPU / driver info"
echo "============================================"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "[WARN] nvidia-smi failed"
nvcc --version 2>/dev/null || echo "[WARN] nvcc not found"

echo
echo "============================================"
echo "Step 2: System python/torch (before venv)"
echo "============================================"
python3 --version
python3 -c "import torch; print('system torch', torch.__version__, '| cuda', torch.version.cuda)" 2>/dev/null || echo "(no system torch)"

echo
echo "============================================"
echo "Step 3: Install ffmpeg"
echo "============================================"
if command -v ffmpeg &>/dev/null; then
    echo "ffmpeg already installed: $(ffmpeg -version 2>&1 | head -1)"
else
    apt-get update -qq && apt-get install -y ffmpeg
fi

echo
echo "============================================"
echo "Step 4: Create venv"
echo "============================================"
python3 -m venv "$VENV"
echo "Venv created at $VENV"
source "$VENV/bin/activate"

echo
echo "============================================"
echo "Step 5: Upgrade pip + install PyTorch cu128"
echo "============================================"
pip install --upgrade pip --quiet
# Pinned to the versions verified working (2026-08-19, and previously in production on
# 2026-08-06): CUDA 12.8 build, required for Blackwell sm_120. torchaudio dropped — nothing
# in this repo imports it.
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

echo
echo "============================================"
echo "Step 6: Verify Blackwell GPU works"
echo "============================================"
python3 -c "
import torch
print('torch:', torch.__version__, '| cuda:', torch.version.cuda)
t = torch.ones(3, 224, 224).cuda()  # will fail if sm_120 unsupported
print('CUDA OK:', torch.cuda.get_device_name(), '| cap:', torch.cuda.get_device_capability())
"

echo
echo "============================================"
echo "Step 7: Install Python packages"
echo "============================================"
pip install -r "$ROOT/requirements.lock.txt"

echo
echo "============================================"
echo "Step 8: Clone CoTracker3"
echo "============================================"
mkdir -p "$REPOS"
# Pinned to the commit already verified in production (see platelock/tracking/track.py in
# the mainline pipeline) so an upstream push can't silently change tracking output here too.
COTRACKER_COMMIT="82e02e8"
if [ ! -d "$REPOS/co-tracker" ]; then
    git clone https://github.com/facebookresearch/co-tracker "$REPOS/co-tracker"
    (cd "$REPOS/co-tracker" && git checkout --quiet "$COTRACKER_COMMIT")
else
    echo "co-tracker already cloned, skipping"
fi
cd "$REPOS/co-tracker" && pip install -e . --no-deps
cd "$ROOT"

echo
echo "============================================"
echo "Setup complete."
echo
echo "Activate with:"
echo "  source $VENV/bin/activate"
echo "============================================"
