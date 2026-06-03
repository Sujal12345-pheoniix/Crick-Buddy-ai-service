#!/usr/bin/env bash
# =============================================================================
# CrickBuddy AI Service — Render Build Script
# Forces CPU-only PyTorch to avoid ~2 GB GPU wheel download on free instances.
# Set "Build Command" in Render dashboard to: bash build.sh
# =============================================================================
set -euo pipefail

echo "🔧 [1/3] Installing CPU-only PyTorch (avoids ~2GB GPU wheel)..."
pip install torch==2.3.1+cpu torchvision==0.18.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu \
    --no-cache-dir \
    --quiet

echo "📦 [2/3] Installing project requirements..."
pip install -r requirements.txt \
    --no-cache-dir \
    --quiet

echo "✅ [3/3] Build complete — CPU-only environment ready"
