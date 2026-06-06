#!/usr/bin/env bash
# Quick host-side setup for enrollment and RKNN export

set -e

echo "================================"
echo "Speaker Enrollment + RKNN Export"
echo "================================"

echo ""
echo "=== STEP 1: Host Dependencies ==="
if command -v python3 >/dev/null 2>&1; then
    echo "Python: $(python3 --version)"
else
    echo "python3 not found"
    exit 1
fi

if [ ! -f "requirements_enrollment.txt" ]; then
    echo "requirements_enrollment.txt not found"
    exit 1
fi

cat <<'INFO'
Install host dependencies with:
  pip install -r requirements_enrollment.txt
INFO

echo ""
echo "=== STEP 2: Train Or Provide A Checkpoint ==="
cat <<'INFO'
Training entry point:
  export LIBRISPEECH_ROOT=/absolute/path/to/LibriSpeech
  python3 main.py

Default checkpoint output:
  output/ECAPATDNN_protonet_model.pth
INFO

echo ""
echo "=== STEP 3: Speaker Enrollment ==="
cat <<'INFO'
Example:
  python3 enroll_lite.py \
      --model-path output/ECAPATDNN_protonet_model.pth \
      --speaker-id alice \
      --audio-files audio_samples/alice/*.wav \
      --store-path output/enrolled_speakers.pt \
      --verbose

Outputs:
  - output/enrolled_speakers.pt

Legacy C/C++ matrix output:
  python3 enroll_lite.py \
      --model-path output/ECAPATDNN_protonet_model.pth \
      --speaker-id alice \
      --audio-files audio_samples/alice/*.wav \
      --store-path output/enrolled_speakers.npy
INFO

echo ""
echo "=== STEP 4: ONNX / RKNN Export ==="
cat <<'INFO'
ONNX only:
  python3 export_to_rknn.py \
      --checkpoint output/ECAPATDNN_protonet_model.pth \
      --onnx-out output/ecapa_tdnn.onnx \
      --onnx-only

RKNN for rv1106:
  python3 export_to_rknn.py \
      --checkpoint output/ECAPATDNN_protonet_model.pth \
      --onnx-out output/ecapa_tdnn.onnx \
      --rknn-out output/ecapa_tdnn.rknn \
      --target-platform rv1106 \
      --quantize \
      --dataset /path/to/calibration_audio_dir
INFO

echo ""
echo "For the full workflow, see QUICKSTART.md."
