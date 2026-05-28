# NeuroSight AI — Eye Fatigue ML Model Training Pipeline

This directory contains the dataset folders, model definition, data preparation utility, and training pipeline to train and export the NeuroSight AI eye fatigue detection model.

---

## 1. Overview
The Eye Fatigue Detection model is a hybrid **CNN + LSTM** deep learning architecture:
- **EyeCNNEncoder**: A lightweight CNN branch utilizing depthwise-separable convolutions to extract visual features from `48x48` grayscale eye crop images.
- **TemporalLSTMEncoder**: A bidirectional LSTM branch with attention pooling to capture temporal progressions of drowsiness metrics (`[EAR, MAR, blink_rate, head_tilt]`) over a sliding window.
- **FatigueClassifier**: Fuses visual and temporal representations to output a probability distribution over **5 drowsiness levels** (`0: Alert`, `1: Mild fatigue`, `2: Moderate fatigue`, `3: Severe fatigue`, `4: Critical / microsleep imminent`).

---

## 2. Directory Structure

```directory
eye-fatigue/
├── drowsiness-detection/      # Primary eye crop dataset (48,000 samples)
│   ├── closed_eye/            # Subfolder of closed eye crop images
│   ├── open_eye/              # Subfolder of open eye crop images
│   └── labels.csv             # Structured metadata labels (auto-generated)
├── mrl-dataset/
│   └── train/                 # Secondary MRL training set (4,000 samples)
│       ├── Closed_Eyes/       # Subfolder of closed eye crop images
│       ├── Open_Eyes/         # Subfolder of open eye crop images
│       └── labels.csv         # Structured metadata labels (auto-generated)
├── notebooks/                 # Scratchpad directory for experiments
├── src/
│   ├── model.py               # Model architecture & PyTorch definitions
│   ├── train.py               # Main pipeline: Dataset, training, validation, & ONNX export
│   └── prepare_data.py        # Dataset labeling & metadata extraction utility
└── README.md                  # This file
```

---

## 3. Environment & Prerequisites

Install the required ML stack using `pip`:

```bash
pip install torch torchvision numpy pandas pillow scikit-learn structlog onnx onnxruntime opencv-python
```

---

## 4. Step-by-Step Training Guide

### Step 4.1: Dataset Labeling & Verification
Before training, run the dataset parser utility to scan the subdirectories, extract metadata from filenames (MRL convention: gender, reflections, lighting, sensor ID, eye state), and write `labels.csv` files:

```bash
python src/prepare_data.py
```

### Step 4.2: Execute the Training Pipeline
Run `src/train.py` to train the model. The pipeline splits the dataset, runs augmentations, optimizes using learning rate schedules, computes multi-class validation metrics (Accuracy, Precision, Recall, F1), and automatically exports the best model to ONNX.

#### Standard Hybrid Training (CNN + LSTM):
```bash
python src/train.py --labels-csv drowsiness-detection/labels.csv mrl-dataset/train/labels.csv --epochs 15 --batch-size 64 --use-focal-loss
```

#### CNN-Only Training (Visual Branch Only):
If you want to train and export *only* the eye image feature extractor branch, pass the `--cnn-only` flag:
```bash
python src/train.py --labels-csv drowsiness-detection/labels.csv --epochs 10 --cnn-only
```

### Step 4.3: Model Training CLI Parameters
Customize the training process using the following command-line flags:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--labels-csv` | `str list` | *Required* | File paths to one or more metadata `labels.csv` files. |
| `--val-split` | `float` | `0.15` | Validation set ratio. |
| `--seq-len` | `int` | `30` | Sequence window length for temporal LSTM. |
| `--cnn-only` | `flag` | `False` | Disables the LSTM branch and trains the CNN branch alone. |
| `--epochs` | `int` | `15` | Number of training epochs. |
| `--batch-size` | `int` | `64` | Training batch size. |
| `--lr` | `float` | `3e-4` | Learning rate (with Cosine Annealing scheduler). |
| `--weight-decay` | `float` | `1e-4` | AdamW L2 weight decay regularization. |
| `--use-focal-loss`| `flag` | `False` | Enables Focal Loss optimization for class imbalances. |
| `--device` | `str` | `cuda` | Target device (`cuda`, `mps`, or `cpu`). |
| `--output-dir` | `str` | `src` | Output directory for `.pt` checkpoint and `.onnx` files. |

---

## 5. Backend Service Deployment

The FastAPI backend model registry (`ModelRegistry`) automatically loads the model at startup and falls back to a lazy-loaded PyTorch model if the optimized ONNX model is missing.

To deploy your trained model to production, copy the generated optimized model file to the configured registry target:

```bash
# Copy ONNX model to register it as the production model
cp src/eye_fatigue.onnx ../../backend/app/resources/models/eye_fatigue.onnx
```

Once copied, the backend API endpoint (`/api/v1/fatigue/metrics` / `/api/v1/fatigue/session/{session_id}`) will instantly run high-performance real-time inference using the new model via ONNX Runtime!
