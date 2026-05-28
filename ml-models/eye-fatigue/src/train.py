"""
NeuroSight AI — Eye Fatigue Model Training Pipeline
Implements dataset loader, training, evaluation, and ONNX export.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import structlog

# Set up logging matching the workspace style
logger = structlog.get_logger(__name__)

# Import model architecture
from model import FatigueClassifier, FocalLoss, FatigueDataAugmentation


# -----------------------------------------------------------
# Joint PyTorch Dataset for Eye Crops & Temporal Features
# -----------------------------------------------------------
class EyeFatigueDataset(Dataset):
    """
    Loads eye crop images and synthesizes corresponding temporal features
    to support training hybrid CNN + LSTM eye-fatigue models.
    """

    def __init__(
        self,
        labels_csv: str | Path,
        transform: transforms.Compose | None = None,
        sequence_length: int = 30,
        synthesize_temporal: bool = True,
    ) -> None:
        self.csv_path = Path(labels_csv)
        self.data_dir = self.csv_path.parent
        self.transform = transform or transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
        ])
        self.seq_len = sequence_length
        self.synthesize_temporal = synthesize_temporal

        # Read CSV file
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Labels file not found at {self.csv_path}")
            
        self.df = pd.read_csv(self.csv_path)
        logger.info(
            "Dataset loaded",
            csv_path=str(self.csv_path),
            total_samples=len(self.df),
            open_eyes=len(self.df[self.df["eye_state"] == 1]),
            closed_eyes=len(self.df[self.df["eye_state"] == 0]),
        )

    def __len__(self) -> int:
        return len(self.df)

    def _synthesize_sequence(self, eye_state: int, target_class: int) -> torch.Tensor:
        """
        Synthesizes a realistic temporal trajectory for [EAR, MAR, blink_rate, head_tilt]
        over a sequence of length T (e.g. 30 frames).
        
        Indices:
          0: EAR (Eye Aspect Ratio) -> normal ~0.33, closed <0.20
          1: MAR (Mouth Aspect Ratio) -> normal ~0.18, yawning >0.60
          2: blink_rate -> normal ~15-20, drowsy decreases/increases
          3: head_tilt -> normal ~0-5 degrees, nodding >15 degrees
        """
        seq = np.zeros((self.seq_len, 4), dtype=np.float32)

        # Baseline features based on drowsiness class target_class (0: Alert -> 4: Critical)
        if target_class == 0:  # Alert
            base_ear = 0.35
            base_mar = 0.18
            base_blink = 18.0
            base_tilt = 2.0
            ear_noise = 0.02
        elif target_class == 1:  # Mild fatigue
            base_ear = 0.32
            base_mar = 0.20
            base_blink = 16.0
            base_tilt = 3.0
            ear_noise = 0.03
        elif target_class == 2:  # Moderate fatigue
            base_ear = 0.28
            base_mar = 0.28
            base_blink = 14.0
            base_tilt = 5.0
            ear_noise = 0.04
        elif target_class == 3:  # Severe fatigue
            base_ear = 0.23
            base_mar = 0.40  # Yawning starts
            base_blink = 10.0
            base_tilt = 12.0
            ear_noise = 0.05
        else:  # Critical / Microsleep imminent
            base_ear = 0.18
            base_mar = 0.55  # Heavy yawning / slack jaw
            base_blink = 6.0
            base_tilt = 22.0  # Head nodding
            ear_noise = 0.06

        for t in range(self.seq_len):
            progress = t / self.seq_len
            
            # Add gradual drift or temporal fluctuation (e.g. micro-sleep event in severe/critical)
            if target_class >= 3 and progress > 0.7:
                # Drowsy event: EAR drops, head tilts, jaw slackens
                ear = base_ear - 0.08 * (progress - 0.7) + np.random.normal(0, ear_noise)
                mar = base_mar + 0.15 * (progress - 0.7) + np.random.normal(0, 0.03)
                tilt = base_tilt + 10.0 * (progress - 0.7) + np.random.normal(0, 2.0)
            else:
                # Normal temporal variance
                ear = base_ear + np.random.normal(0, ear_noise)
                mar = base_mar + np.random.normal(0, 0.02)
                tilt = base_tilt + np.random.normal(0, 1.0)

            # Ensure final eye state constraints are strictly met
            if t == self.seq_len - 1:
                if eye_state == 0:
                    ear = min(ear, 0.20)  # Must look closed
                else:
                    ear = max(ear, 0.26)  # Must look open

            seq[t, 0] = np.clip(ear, 0.05, 0.45)
            seq[t, 1] = np.clip(mar, 0.05, 0.90)
            seq[t, 2] = np.clip(base_blink, 2.0, 30.0) / 30.0  # Normalize blink rate
            seq[t, 3] = np.clip(tilt, 0.0, 45.0) / 45.0        # Normalize tilt angle

        return torch.from_numpy(seq)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor | int, int]:
        row = self.df.iloc[idx]
        img_path = self.data_dir / row["filename"]
        eye_state = int(row["eye_state"])

        # Load image crop as grayscale (L mode)
        try:
            with Image.open(img_path) as img:
                img_tensor = self.transform(img.convert("L"))
        except Exception as e:
            logger.warning("Failed to load image, using empty tensor", path=str(img_path), error=str(e))
            img_tensor = torch.zeros((1, 48, 48), dtype=torch.float32)

        # Map binary eye state label into 5-class target:
        # Open (1) maps to Class 0 (Alert) or Class 1 (Mild Fatigue)
        # Closed (0) maps to Class 3 (Severe) or Class 4 (Critical)
        if eye_state == 1:
            target_class = random.choices([0, 1, 2], weights=[0.75, 0.20, 0.05], k=1)[0]
        else:
            target_class = random.choices([2, 3, 4], weights=[0.05, 0.25, 0.70], k=1)[0]

        if self.synthesize_temporal:
            temporal_tensor = self._synthesize_sequence(eye_state, target_class)
            return img_tensor, temporal_tensor, target_class
        
        return img_tensor, target_class


# -----------------------------------------------------------
# Model Training Pipeline Manager
# -----------------------------------------------------------
class FatigueModelTrainer:

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Training device configuration", device=str(self.device))

    def prepare_data(self) -> tuple[DataLoader, DataLoader]:
        # Gather all label CSVs
        csv_paths = [Path(p) for p in self.args.labels_csv]
        datasets = []

        # Standard image augmentations for training
        train_img_transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
        ])

        for path in csv_paths:
            dataset = EyeFatigueDataset(
                labels_csv=path,
                transform=train_img_transform,
                sequence_length=self.args.seq_len,
                synthesize_temporal=(not self.args.cnn_only),
            )
            datasets.append(dataset)

        # Concat multiple datasets if provided
        if len(datasets) > 1:
            full_dataset = torch.utils.data.ConcatDataset(datasets)
        else:
            full_dataset = datasets[0]

        # Train / Validation Split
        val_size = int(len(full_dataset) * self.args.val_split)
        train_size = len(full_dataset) - val_size
        
        # Fixing seeds for reproducible splits
        generator = torch.Generator().manual_seed(42)
        train_subset, val_subset = random_split(
            full_dataset, [train_size, val_size], generator=generator
        )

        # Apply basic transform (no augmentation) for validation subset
        val_subset.dataset.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
        ])

        train_loader = DataLoader(
            train_subset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

        logger.info(
            "Data loaders initialized",
            train_samples=len(train_subset),
            val_samples=len(val_subset),
            batch_size=self.args.batch_size,
        )
        return train_loader, val_loader

    def train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Any,
        augmentation: FatigueDataAugmentation,
    ) -> float:
        model.train()
        total_loss = 0.0

        for batch in loader:
            optimizer.zero_grad()

            if self.args.cnn_only:
                img, target = batch
                img = img.to(self.device)
                img = augmentation(img)
                target = target.to(self.device)
                out = model(eye_image=img)
            else:
                img, seq, target = batch
                img = img.to(self.device)
                img = augmentation(img)
                seq = seq.to(self.device)
                target = target.to(self.device)
                out = model(eye_image=img, temporal_features=seq)

            loss = criterion(out["logits"], target)
            loss.backward()
            
            # Gradient clipping to prevent gradient explosion in LSTM
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()

        if scheduler:
            scheduler.step()

        return total_loss / len(loader)

    @torch.inference_mode()
    def evaluate(self, model: nn.Module, loader: DataLoader, criterion: nn.Module) -> dict[str, float]:
        model.eval()
        total_loss = 0.0
        
        all_preds = []
        all_targets = []

        for batch in loader:
            if self.args.cnn_only:
                img, target = batch
                img = img.to(self.device)
                target = target.to(self.device)
                out = model(eye_image=img)
            else:
                img, seq, target = batch
                img = img.to(self.device)
                seq = seq.to(self.device)
                target = target.to(self.device)
                out = model(eye_image=img, temporal_features=seq)

            loss = criterion(out["logits"], target)
            total_loss += loss.item()

            preds = out["predicted_class"].cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(target.cpu().numpy())

        accuracy = accuracy_score(all_targets, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average="macro", zero_division=0
        )

        return {
            "loss": total_loss / len(loader),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }

    def export_onnx(self, model: nn.Module, save_path: Path) -> None:
        """
        Exports the model to ONNX with dynamic batch axes
        so it is production-ready for ONNX Runtime.
        """
        logger.info("Exporting model to ONNX format...")
        model.eval()
        model.to("cpu")

        if self.args.cnn_only:
            dummy_img = torch.randn(1, 1, 48, 48)
            torch.onnx.export(
                model,
                (dummy_img, None),
                str(save_path),
                input_names=["eye_image"],
                output_names=["logits", "probabilities", "fatigue_score", "predicted_class"],
                dynamic_axes={"eye_image": {0: "batch"}},
                opset_version=17,
            )
        else:
            dummy_img = torch.randn(1, 1, 48, 48)
            dummy_seq = torch.randn(1, self.args.seq_len, 4)
            torch.onnx.export(
                model,
                (dummy_img, dummy_seq),
                str(save_path),
                input_names=["eye_image", "temporal_features"],
                output_names=["logits", "probabilities", "fatigue_score", "predicted_class"],
                dynamic_axes={"eye_image": {0: "batch"}, "temporal_features": {0: "batch"}},
                opset_version=17,
            )

        logger.info("ONNX export complete", path=str(save_path))

    def run(self) -> None:
        logger.info("Starting Eye Fatigue ML Model training pipeline...")

        # 1. Prepare data loaders
        train_loader, val_loader = self.prepare_data()

        # 2. Instantiate hybrid Fatigue Classifier Model
        use_cnn = True
        use_lstm = not self.args.cnn_only
        
        model = FatigueClassifier(
            use_cnn=use_cnn,
            use_lstm=use_lstm,
            dropout=self.args.dropout
        ).to(self.device)

        # Data augmentation block
        augmentation = FatigueDataAugmentation(training=True).to(self.device)

        # 3. Optimization Setup
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )
        
        # Cosine learning rate decay scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.args.epochs,
            eta_min=1e-6
        )

        # Class imbalance optimization
        if self.args.use_focal_loss:
            criterion = FocalLoss(alpha=1.0, gamma=2.0)
            logger.info("Using Focal Loss objective")
        else:
            criterion = nn.CrossEntropyLoss()
            logger.info("Using Standard Cross-Entropy objective")

        best_val_f1 = 0.0
        best_pt_path = self.output_dir / "best_model.pt"

        # 4. Main Training / Validation Epoch Loop
        for epoch in range(self.args.epochs):
            train_loss = self.train_epoch(
                model, train_loader, criterion, optimizer, scheduler, augmentation
            )
            val_metrics = self.evaluate(model, val_loader, criterion)

            logger.info(
                f"Epoch {epoch+1:02d}/{self.args.epochs:02d}",
                train_loss=round(train_loss, 4),
                val_loss=round(val_metrics["loss"], 4),
                val_acc=round(val_metrics["accuracy"] * 100, 2),
                val_f1=round(val_metrics["f1_score"] * 100, 2),
            )

            # Checkpoint saving on improvement
            if val_metrics["f1_score"] > best_val_f1:
                best_val_f1 = val_metrics["f1_score"]
                torch.save(model.state_dict(), best_pt_path)
                logger.info(
                    "Saved improved checkpoint",
                    path=str(best_pt_path),
                    f1_score=round(best_val_f1 * 100, 2),
                )

        logger.info("Training complete. Loading best parameters for deployment export...")
        
        # Load best weights
        if best_pt_path.exists():
            model.load_state_dict(torch.load(best_pt_path))
            
        # Export final best model to ONNX for registry
        onnx_save_path = self.output_dir / "eye_fatigue.onnx"
        self.export_onnx(model, onnx_save_path)

        logger.info("All pipeline processes completed successfully!")


# -----------------------------------------------------------
# CLI Argument Parser Setup
# -----------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NeuroSight AI — Eye Fatigue Classifier Training Pipeline"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--labels-csv",
        type=str,
        nargs="+",
        required=True,
        help="Paths to one or more metadata labels.csv files (whitespace separated)"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Ratio of data allocated to the validation set"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=30,
        help="Temporal feature sequence length for the LSTM branch"
    )
    parser.add_argument(
        "--cnn-only",
        action="store_true",
        help="Disable the temporal LSTM branch and train using only CNN visual crops"
    )

    # Hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Loader batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.4,
        help="Dropout probability in classifier heads"
    )
    parser.add_argument(
        "--use-focal-loss",
        action="store_true",
        help="Enable Focal Loss to combat class imbalance"
    )

    # Infrastructure
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Computation device ('cuda', 'mps', or 'cpu')"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers (0 for Windows/safety)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="src",
        help="Directory to save output checkpoints and ONNX files"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    FatigueModelTrainer(args).run()
