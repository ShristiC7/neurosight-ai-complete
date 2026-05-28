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