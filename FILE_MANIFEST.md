# File Manifest

## Kept for Current Workflow

### Enrollment
- `enroll_lite.py`: host-side speaker enrollment that writes `enrolled_speakers.pt` by default
- `requirements_enrollment.txt`: host-side Python dependencies for enrollment

### RKNN Conversion
- `export_to_rknn.py`: convert checkpoint or ONNX model to RKNN
- `prepare_rknn_calibration_dataset.py`: build `.npy` calibration data for quantization

### Model and Utilities
- `model/`: ECAPA-TDNN model definitions
- `utils/`: audio preprocessing and model loading helpers

### Training / Experiments
- `main.py`
- `PrototypicalNetwork/`

## Removed from This Repo
- Python inference scripts
- Python deployment helpers
- Python runtime wrappers for RKNN
- device-only Python requirement files

## Enrollment Output Format
- `enrolled_speakers.pt`: default speaker store keyed by speaker ID
- `enrolled_speakers.npy`: optional legacy `float32` matrix of shape `(num_speakers, embedding_dim)`
- `enrolled_speakers_ids.txt`: sidecar IDs file used only with the legacy `.npy` format
