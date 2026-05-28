# preprocess.py
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import argparse

def synthesize_sequence(img_path: Path, eye_state: int) -> np.ndarray:
    """
    Very simple placeholder: generate a constant 30‑step sequence.
    Replace with the full physics‑based synthesis from your original repo
    if you want richer temporal features.
    """
    seq = np.zeros((30, 4), dtype=np.float32)

    # EAR (Eye Aspect Ratio)
    seq[:, 0] = 0.35 if eye_state == 1 else 0.18
    # MAR (Mouth Aspect Ratio) – dummy constant
    seq[:, 1] = 0.18
    # Blink rate – normalized dummy
    seq[:, 2] = 0.5
    # Head tilt – dummy
    seq[:, 3] = 0.1
    return seq

def main(data_dir: Path):
    # All CSVs under the data dir (recursive)
    csv_files = list(data_dir.rglob("labels.csv"))
    all_seqs = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            img_path = data_dir / row["filename"]
            eye_state = int(row["eye_state"])
            seq = synthesize_sequence(img_path, eye_state)
            all_seqs.append(seq)
    all_seqs = np.stack(all_seqs)  # shape (N, 30, 4)
    out_path = data_dir / "processed" / "train_sequences.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, all_seqs)
    print(f"Saved {all_seqs.shape[0]} sequences → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Root folder that contains the image crops and label CSVs")
    args = parser.parse_args()
    main(args.data_dir)
