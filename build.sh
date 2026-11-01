#!/usr/bin/env bash
# =============================================================================
# CrickBuddy AI Service — Render Build Script
# Set "Build Command" in Render dashboard to: bash build.sh
# (plain `pip install -r requirements.txt` works too — PyTorch is no longer needed)
# =============================================================================
set -euo pipefail

echo "📦 [1/2] Installing requirements..."
pip install -r requirements.txt --no-cache-dir --quiet

echo "🔍 [2/2] Checking models and pose API..."
test -f models/yolov8n.onnx && test -f models/yolov8n-pose.onnx
python -c "import mediapipe.python.solutions.pose, onnxruntime; print('mediapipe + onnxruntime OK')"

echo "✅ Build complete"
