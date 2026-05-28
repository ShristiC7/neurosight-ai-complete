# NeuroSight AI — Next Steps Guide

## Part 2 of 3: Sample Datasets & Model Training

> This section explains exactly what data each model uses, where to get real datasets,
> how to run training, and what output files to expect.

---

## Overview of All 5 Models

| Model | Training Data Source | Training Time | Output File |
|---|---|---|---|
| Eye Fatigue (CNN+LSTM) | MRL Eye Dataset + synthetic | ~20 min on CPU | `eye-fatigue/model.onnx` |
| Voice Stress (Transformer) | RAVDESS + CREMA-D | ~45 min on CPU | `voice-stress/model.onnx` |
| Behavioral Analytics (Autoencoder + IF) | Synthetic (auto-generated) | ~3 min on CPU | `behavioral-analytics/autoencoder.onnx` |
| Productivity Forecasting (LSTM + XGBoost) | Synthetic (auto-generated) | ~5 min on CPU | `productivity-predictor/src/xgboost.json` |
| RL Recommendation Agent (DQN) | Trains online from user feedback | Continuous | `rl-agent/agent.zip` |

---

## 2.1 Model 1 — Eye Fatigue Detection

### What It Learns
The model classifies eye states into 5 drowsiness levels (alert → critical) using:
- 48×48 grayscale eye crop images
- Time-series sequences of Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), blink rate, head tilt

### Sample Data Structure

```
ml-models/eye-fatigue/data/
├── raw/
│   ├── open_eyes/          ← images of alert eyes
│   │   ├── subject01_001.jpg
│   │   ├── subject01_002.jpg
│   │   └── ...
│   ├── closed_eyes/        ← images of closed eyes
│   ├── sleepy_eyes/        ← images of partially closed eyes
│   └── labels.csv          ← image_path, label (0-4), ear_value
│
└── processed/
    ├── train_sequences.npy  ← (N, 30, 4) temporal sequences
    ├── train_images.npy     ← (N, 1, 48, 48) eye crops
    └── train_labels.npy     ← (N,) class labels 0-4
```

### Where to Get Real Training Data

**Option A — Free public datasets (recommended to start):**

| Dataset | Size | Download | What It Contains |
|---|---|---|---|
| **MRL Eye Dataset** | 84,898 images | [mrl.cs.vsb.cz/eyedataset](http://mrl.cs.vsb.cz/eyedataset) | Open/closed eyes under different conditions |
| **ZJU Eyeblink** | 80 video sequences | [research request](http://www.esi.zju.edu.cn) | Natural blink sequences with EAR labels |
| **Closed Eyes in the Wild (CEW)** | 2,423 subjects | [parnec.nuaa.edu.cn](http://parnec.nuaa.edu.cn/xtan/data/ClosedEyeDatabases.html) | Closed eye detection |
| **NTHU Drowsy Driver** | ~36GB video | [nthu-en.nthu.edu.tw](http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/) | Full drowsiness video (needs account) |

**Option B — Kaggle (easiest access):**

```bash
# Install Kaggle CLI
pip install kaggle

# Set up credentials: https://www.kaggle.com/docs/api
# Download directly from Kaggle
kaggle datasets download -d prasadvpatil/mrl-dataset
kaggle datasets download -d kutaykutlu/drowsiness-detection
```

> 💡 **Start with MRL + Kaggle drowsiness dataset.** Together they give you ~100k images covering all drowsiness levels — enough to train a reliable model.

### Sample `labels.csv` Format

```csv
image_path,label,ear_value,subject_id,session_id
open_eyes/s001_f0001.jpg,0,0.38,s001,morning_1
open_eyes/s001_f0002.jpg,0,0.36,s001,morning_1
sleepy_eyes/s001_f0180.jpg,2,0.22,s001,morning_1
closed_eyes/s001_f0210.jpg,4,0.08,s001,morning_1
```

Label mapping:
- `0` = alert (EAR > 0.35)
- `1` = mild fatigue (EAR 0.28–0.35)
- `2` = moderate (EAR 0.20–0.28)
- `3` = severe (EAR 0.12–0.20)
- `4` = critical / microsleep (EAR < 0.12)

### How to Create a Training Script

Create `ml-models/eye-fatigue/src/train.py`:

```python
"""
Eye Fatigue Model Training Script.
Run: python train.py --data-dir ../data/raw --epochs 50
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from model import FatigueClassifier, FocalLoss, FatigueDataAugmentation


class EyeDataset(Dataset):
    """Loads eye images + temporal sequences for training."""

    def __init__(self, image_paths, sequences, labels, augment=False):
        self.image_paths = image_paths
        self.sequences = sequences       # (N, 30, 4) temporal features
        self.labels = labels
        self.augment = FatigueDataAugmentation(training=augment)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Load and preprocess eye image
        img = cv2.imread(str(self.image_paths[idx]), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48, 48))
        img = torch.from_numpy(img).float().unsqueeze(0) / 255.0  # (1, 48, 48)
        img = self.augment(img)

        seq = torch.from_numpy(self.sequences[idx]).float()  # (30, 4)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return img, seq, label


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # --- Load data ---
    df = pd.read_csv(args.data_dir / "labels.csv")
    sequences = np.load(args.data_dir / "processed/train_sequences.npy")

    X_train, X_val, seq_train, seq_val, y_train, y_val = train_test_split(
        df["image_path"].values,
        sequences,
        df["label"].values,
        test_size=0.2,
        stratify=df["label"].values,
        random_state=42,
    )

    train_ds = EyeDataset(X_train, seq_train, y_train, augment=True)
    val_ds   = EyeDataset(X_val,   seq_val,   y_val,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=4)

    # --- Model ---
    model = FatigueClassifier().to(device)
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Training loop ---
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for imgs, seqs, labels in train_loader:
            imgs, seqs, labels = imgs.to(device), seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(eye_image=imgs, temporal_features=seqs)
            loss = criterion(output["logits"], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, seqs, labels in val_loader:
                imgs, seqs, labels = imgs.to(device), seqs.to(device), labels.to(device)
                output = model(eye_image=imgs, temporal_features=seqs)
                preds = output["predicted_class"]
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2%}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output_dir / "model_best.pt")
            print(f"  ↑ New best saved ({val_acc:.2%})")

    # Export to ONNX
    print("Exporting to ONNX...")
    model.load_state_dict(torch.load(args.output_dir / "model_best.pt"))
    model.eval()

    dummy_img = torch.randn(1, 1, 48, 48)
    dummy_seq = torch.randn(1, 30, 4)
    torch.onnx.export(
        model,
        {"eye_image": dummy_img, "temporal_features": dummy_seq},
        args.output_dir / "model.onnx",
        input_names=["eye_image", "temporal_features"],
        output_names=["logits", "probabilities", "fatigue_score", "predicted_class"],
        dynamic_axes={"eye_image": {0: "batch"}, "temporal_features": {0: "batch"}},
        opset_version=17,
    )
    print(f"Done! Model saved to {args.output_dir}/model.onnx")
    print(f"Best validation accuracy: {best_val_acc:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   type=Path, default=Path("../data"))
    parser.add_argument("--output-dir", type=Path, default=Path("../"))
    parser.add_argument("--epochs",     type=int,  default=50)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train(args)
```

### Run Training
```bash
cd ml-models/eye-fatigue/src
python train.py --data-dir ../data --epochs 50
# Output: ml-models/eye-fatigue/model.onnx
```

Expected results:
- Training time: ~20 min on CPU, ~3 min on GPU
- Val accuracy target: >92%
- Output file size: ~8MB ONNX

---

## 2.2 Model 2 — Voice Stress Detection

### What It Learns
Classifies audio into 5 emotion states (calm/stressed/fatigued/energetic/anxious) and predicts a continuous stress score, from mel spectrograms + MFCC features.

### Where to Get Training Data

| Dataset | Size | Download | License |
|---|---|---|---|
| **RAVDESS** | 24 actors, 1,440 clips | [zenodo.org/record/1188976](https://zenodo.org/record/1188976) | CC BY-NC-SA |
| **CREMA-D** | 7,442 clips, 91 actors | [github.com/CheyneyComputerScience/CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) | Open |
| **SAVEE** | 480 clips, 4 actors | [surrey.ac.uk/Personal/ss/...](http://kahlan.eps.surrey.ac.uk/savee/) | Research only |
| **ESD** (Emotional Speech Dataset) | 350 parallel sentences | [github.com/HLTSingapore/Emotional-Speech-Data](https://github.com/HLTSingapore/Emotional-Speech-Data) | MIT |

**Download RAVDESS + CREMA-D (recommended combination):**

```bash
# RAVDESS — direct download
mkdir -p ml-models/voice-stress/data/raw/ravdess
cd ml-models/voice-stress/data/raw/ravdess
wget https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip
unzip Audio_Speech_Actors_01-24.zip

# CREMA-D — via git LFS
git clone https://github.com/CheyneyComputerScience/CREMA-D.git
# AudioWAV/ contains the .wav files
```

### RAVDESS Emotion Label Mapping

RAVDESS filenames encode the emotion in position 3:
```
03-01-05-01-02-01-12.wav
         ↑
         Emotion code: 01=neutral, 02=calm, 03=happy, 04=sad,
                       05=angry, 06=fearful, 07=disgust, 08=surprised
```

Our mapping:
```python
RAVDESS_TO_NEUROSIGHT = {
    "01": "calm",       # neutral → calm
    "02": "calm",       # calm
    "03": "energetic",  # happy → energetic
    "04": "fatigued",   # sad → fatigued
    "05": "stressed",   # angry → stressed
    "06": "anxious",    # fearful → anxious
    "07": "stressed",   # disgust → stressed
    "08": "energetic",  # surprised → energetic
}
```

### Sample Data Structure

```
ml-models/voice-stress/data/
├── raw/
│   ├── ravdess/
│   │   ├── Actor_01/
│   │   │   ├── 03-01-01-01-01-01-01.wav
│   │   │   └── ...
│   │   └── Actor_24/
│   └── crema-d/
│       ├── AudioWAV/
│       │   ├── 1001_DFA_ANG_XX.wav
│       │   └── ...
│       └── VideoDemographics.csv
│
└── processed/
    ├── spectrograms/    ← (N, 1, 128, T) mel spectrograms as .npy
    ├── mfcc_stats.npy   ← (N, 39) MFCC + delta statistics
    └── labels.csv       ← file_path, emotion, stress_score
```

### Preprocessing Script

Create `ml-models/voice-stress/src/preprocess.py`:

```python
"""
Preprocess RAVDESS + CREMA-D into spectrograms + MFCC features.
Run: python preprocess.py --data-dir ../data/raw --output-dir ../data/processed
"""

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


RAVDESS_EMOTION_MAP = {
    "01": ("calm", 15.0),
    "02": ("calm", 10.0),
    "03": ("energetic", 55.0),
    "04": ("fatigued", 60.0),
    "05": ("stressed", 80.0),
    "06": ("anxious", 75.0),
    "07": ("stressed", 70.0),
    "08": ("energetic", 50.0),
}

CREMA_EMOTION_MAP = {
    "ANG": ("stressed", 80.0),
    "DIS": ("stressed", 65.0),
    "FEA": ("anxious",  78.0),
    "HAP": ("energetic", 50.0),
    "NEU": ("calm",     15.0),
    "SAD": ("fatigued", 60.0),
}

SR = 22050          # Sample rate
N_MELS = 128        # Mel bins
N_MFCC = 13         # MFCC coefficients
TARGET_DURATION = 3.0  # Seconds — pad/truncate all clips to 3s


def extract_features(audio_path: Path):
    y, _ = librosa.load(audio_path, sr=SR, duration=TARGET_DURATION)

    # Pad or truncate
    target_samples = int(TARGET_DURATION * SR)
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)))
    else:
        y = y[:target_samples]

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)

    # MFCC + delta + delta-delta (39 features total)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    mfcc_stats = np.concatenate([
        mfcc.mean(axis=1),       # 13 features
        mfcc_delta.mean(axis=1), # 13 features
        mfcc_delta2.mean(axis=1) # 13 features
    ])  # → 39-dim vector

    return mel_norm, mfcc_stats


def preprocess(args):
    records = []
    spectrograms = []
    mfcc_list = []

    # Process RAVDESS
    for wav in tqdm(sorted(Path(args.data_dir / "ravdess").rglob("*.wav")), desc="RAVDESS"):
        parts = wav.stem.split("-")
        if len(parts) < 3:
            continue
        emotion_code = parts[2]
        if emotion_code not in RAVDESS_EMOTION_MAP:
            continue

        emotion, stress_score = RAVDESS_EMOTION_MAP[emotion_code]
        mel, mfcc_stats = extract_features(wav)

        spectrograms.append(mel[np.newaxis, ...])  # add channel dim
        mfcc_list.append(mfcc_stats)
        records.append({"file": str(wav), "emotion": emotion, "stress_score": stress_score})

    # Process CREMA-D
    for wav in tqdm(sorted(Path(args.data_dir / "crema-d/AudioWAV").glob("*.wav")), desc="CREMA-D"):
        parts = wav.stem.split("_")
        if len(parts) < 3:
            continue
        emotion_code = parts[2]
        if emotion_code not in CREMA_EMOTION_MAP:
            continue

        emotion, stress_score = CREMA_EMOTION_MAP[emotion_code]
        mel, mfcc_stats = extract_features(wav)

        spectrograms.append(mel[np.newaxis, ...])
        mfcc_list.append(mfcc_stats)
        records.append({"file": str(wav), "emotion": emotion, "stress_score": stress_score})

    # Save
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    np.save(output / "spectrograms.npy",  np.array(spectrograms, dtype=np.float32))
    np.save(output / "mfcc_stats.npy",    np.array(mfcc_list,    dtype=np.float32))
    pd.DataFrame(records).to_csv(output / "labels.csv", index=False)

    print(f"\nDone! Processed {len(records)} audio files")
    print(f"Spectrogram shape: {np.array(spectrograms).shape}")
    print(f"MFCC stats shape:  {np.array(mfcc_list).shape}")
    print(f"Saved to: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   type=Path, default=Path("../data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("../data/processed"))
    preprocess(parser.parse_args())
```

### Run Training

```bash
cd ml-models/voice-stress/src

# Step 1 — Preprocess audio data
python preprocess.py \
  --data-dir ../data/raw \
  --output-dir ../data/processed

# Step 2 — Train model (training script in the codebase already has the architecture)
# Add train.py following the same pattern as eye-fatigue/src/train.py
# Expected: Val F1 > 0.85, training ~45 min on CPU
```

---

## 2.3 Model 3 — Behavioral Analytics (Auto-Generated Data)

This model **does not need real-world data** to get started. It generates synthetic behavioral data internally.

### Run Training (Zero Setup)

```bash
cd ml-models/behavioral-analytics/src
python train.py
```

**What happens:**
1. Generates 5,000 samples of "normal" work behavior
2. Generates 500 samples of "anomalous" (fatigued) behavior
3. Trains Isolation Forest (outlier detection)
4. Trains Autoencoder (256-dim embedding generator)
5. Evaluates and saves models

**Output files:**
```
ml-models/behavioral-analytics/
├── isolation_forest.pkl    ← Scikit-learn model
├── scaler.pkl              ← StandardScaler for features
└── autoencoder.onnx        ← ONNX autoencoder (~2MB)
```

### Sample Synthetic Data (what it generates internally)

| Feature | Normal Range | Anomalous Range |
|---|---|---|
| Typing Speed (WPM) | 40–65 | 8–28 |
| Typing Rhythm Variance (ms) | 60–180 | 250–600 |
| Error Rate (per min) | 0.5–2.0 | 5.0–15.0 |
| Mouse Entropy (0–1) | 0.5–0.8 | 0.1–0.25 |
| App Switch Frequency (per hr) | 3–12 | 25–50 |
| Focus Duration (min) | 15–90 | 1–8 |
| Idle Time (sec/10min) | 20–90 | 200–500 |

### Improve With Real Data Later

Once the app is running and users log in, real behavioral data accumulates in your database. Retrain with real data:

```python
import pandas as pd
import psycopg2

# Export from your PostgreSQL database
conn = psycopg2.connect("postgresql://neurosight:password@localhost/neurosight")
df = pd.read_sql("""
    SELECT typing_speed, typing_rhythm_variance, error_rate,
           mouse_movement_entropy, app_switch_frequency,
           focus_session_duration, idle_time, behavior_score
    FROM behavioral_metrics
    WHERE created_at > NOW() - INTERVAL '30 days'
""", conn)

df.to_csv("ml-models/behavioral-analytics/data/real_data.csv", index=False)
# Then re-run: python train.py --data-path ../data/real_data.csv
```

---

## 2.4 Model 4 — Productivity Forecasting (Auto-Generated Data)

Same as behavioral analytics — no external data needed to start.

### Run Training (Zero Setup)

```bash
cd ml-models/productivity-predictor/src
python train.py
```

**What happens:**
1. Generates 2,000 synthetic work sessions × 30 time steps each
2. Trains Bidirectional LSTM with attention
3. Trains XGBoost regressor on snapshot features
4. Evaluates ensemble (target: RMSE < 10 points on 0–100 scale)
5. Exports ONNX + saves XGBoost model

**Output files:**
```
ml-models/productivity-predictor/src/
├── xgboost.json    ← XGBoost model (~500KB)
└── lstm.onnx       ← LSTM model (~12MB)
```

### Synthetic Session Types Generated

| Session Type | Base Productivity | Fatigue Drift | Burnout Probability |
|---|---|---|---|
| Peak | 75–95% | Low (0–5%) | 0–20% |
| Normal | 50–75% | Moderate (2–10%) | 10–40% |
| Declining | 30–60% | High (5–20%) | 40–70% |
| Exhausted | 10–35% | Very High (15–35%) | 70–100% |

---

## 2.5 Model 5 — RL Recommendation Agent

The RL agent **trains online** — it starts with random weights and learns from users accepting/rejecting recommendations.

### Initial State (Day 1)

No training needed. The agent:
- Starts with epsilon = 1.0 (100% random recommendations)
- Gradually shifts to learned recommendations as epsilon decays
- After ~1,000 interactions, recommendations become meaningfully personalized

### Accelerate Training With Simulated Interactions

If you want faster convergence before real users, run the simulation:

Create `ml-models/rl-agent/src/simulate.py`:

```python
"""
Pre-train the RL agent on simulated user interactions.
Run: python simulate.py --steps 50000
"""

import argparse
import numpy as np
from agent import ProductivityRLAgent, ACTION_NAMES


def simulate_user_response(state: np.ndarray, action_name: str) -> float:
    """Simulate whether a simulated user would accept this recommendation."""
    fatigue = state[0]
    stress = state[1]
    productivity = state[2]
    burnout = state[4]

    # Fatigue is high → user is likely to accept break/eye_rest
    if fatigue > 0.7 and action_name in ("take_break", "eye_rest", "stretch"):
        return np.random.choice([0.8, 0.5], p=[0.8, 0.2])
    # Productivity is high → user rejects break suggestions
    if productivity > 0.8 and action_name == "take_break":
        return np.random.choice([-0.3, 0.5], p=[0.7, 0.3])
    # Default acceptance probability
    return np.random.choice([0.5, -0.3], p=[0.55, 0.45])


def run_simulation(steps: int):
    agent = ProductivityRLAgent()
    rng = np.random.default_rng(42)

    print(f"Simulating {steps:,} interactions...")

    for step in range(steps):
        # Random state vector (12-dim)
        state = rng.uniform(0, 1, size=12).astype(np.float32)

        # Agent picks action
        action = agent.select_action(state, user_id="simulation")

        # Simulate next state (small improvement after accepted action)
        next_state = state.copy()
        next_state[0] = max(0, state[0] - rng.uniform(0, 0.1))  # fatigue decreases

        # Simulate reward
        reward = simulate_user_response(state, ACTION_NAMES[action])

        # Store and train
        agent.store_transition(state, action, reward, next_state, done=False)
        loss = agent.train_step()

        if (step + 1) % 5000 == 0:
            print(f"  Step {step+1:,} | ε={agent.epsilon:.3f} | Loss={loss:.4f if loss else 'N/A'}")

    agent.save("../agent.zip")
    print(f"\nPre-training complete! Saved to ml-models/rl-agent/agent.zip")
    print(f"Final epsilon: {agent.epsilon:.3f} (lower = more learned behavior)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50000)
    run_simulation(parser.parse_args().steps)
```

```bash
cd ml-models/rl-agent/src
python simulate.py --steps 50000
# Takes ~2 minutes. Output: ml-models/rl-agent/agent.zip
```

---

## 2.6 Complete Training Order

Run models in this order (dependencies flow downward):

```
Step 1 — No dependencies
├── python ml-models/behavioral-analytics/src/train.py      (~3 min)
└── python ml-models/productivity-predictor/src/train.py    (~5 min)

Step 2 — Needs datasets downloaded first
├── python ml-models/eye-fatigue/src/train.py               (~20 min)
└── python ml-models/voice-stress/src/preprocess.py
    python ml-models/voice-stress/src/train.py              (~45 min)

Step 3 — Run after app has data, or simulate
└── python ml-models/rl-agent/src/simulate.py               (~2 min)
```

**Minimum to get the app running:** Only Steps 1 and 3 are required. The eye fatigue and voice stress models fall back to heuristic scoring if their ONNX files are missing.

---

## 2.7 Verify Models Are Loadable

```bash
cd backend
python3 -c "
import asyncio, sys
sys.path.insert(0, '.')
from app.core.config import settings
from app.services.ml_registry import ModelRegistry

async def check():
    await ModelRegistry.initialize()
    loaded = ModelRegistry.loaded_models()
    print('Loaded models:', loaded)
    if not loaded:
        print('WARNING: No models loaded. App will use heuristics only.')

asyncio.run(check())
"
```

Expected output (after running all training):
```
Loaded models: ['eye_fatigue', 'voice_stress', 'productivity', 'rl_agent']
```

---

*Next: Part 3 — Testing Guide →*
