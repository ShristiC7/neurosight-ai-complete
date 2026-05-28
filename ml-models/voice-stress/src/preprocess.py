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