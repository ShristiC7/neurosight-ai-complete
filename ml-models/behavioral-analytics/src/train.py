"""
NeuroSight AI — Behavioral Analytics Training Pipeline
Isolation Forest + Autoencoder for anomaly detection.
Builds the behavioral embedding space stored in Qdrant.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import structlog

logger = structlog.get_logger(__name__)

# -----------------------------------------------------------
# Autoencoder — Behavioral Anomaly Detector
# -----------------------------------------------------------
class BehavioralAutoencoder(nn.Module):
    """
    Autoencoder for behavioral anomaly detection.
    Normal behavior → low reconstruction error
    Anomalous behavior → high reconstruction error

    Used alongside Isolation Forest for robust detection.
    Also generates 256-dim behavioral embeddings for Qdrant.

    Input: 10-dimensional behavioral feature vector
    Bottleneck: 256-dim embedding (used as Qdrant vector)
    """

    INPUT_DIM = 10
    EMBEDDING_DIM = 256

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()

        # Encoder: 10 → 64 → 128 → 256
        self.encoder = nn.Sequential(
            nn.Linear(self.INPUT_DIM, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(128, self.EMBEDDING_DIM),
            nn.LayerNorm(self.EMBEDDING_DIM),
        )

        # Decoder: 256 → 128 → 64 → 10
        self.decoder = nn.Sequential(
            nn.Linear(self.EMBEDDING_DIM, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),

            nn.Linear(64, self.INPUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder(x)

    def get_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Returns per-sample reconstruction error as anomaly score."""
        with torch.no_grad():
            recon, _ = self.forward(x)
            return torch.mean((x - recon) ** 2, dim=-1)


# -----------------------------------------------------------
# Feature Engineering for Behavioral Data
# -----------------------------------------------------------
def build_behavioral_features(metrics: dict) -> np.ndarray:
    """
    Build the 10-dimensional behavioral feature vector from raw metrics.

    Features:
    0: typing_speed_normalized     (WPM / 100)
    1: typing_rhythm_stability     (1 - variance / 1000)
    2: error_rate_normalized       (errors_per_min / 20)
    3: mouse_entropy               (0-1)
    4: mouse_click_rate_norm       (clicks/min / 60)
    5: app_switch_frequency_norm   (switches/hr / 40)
    6: focus_duration_norm         (minutes / 120)
    7: idle_time_ratio             (idle_secs / 600)
    8: behavior_score_norm         (0-1)
    9: time_of_day_normalized      (hour / 24)
    """
    return np.array([
        min(metrics.get("typing_speed", 0) / 100.0, 1.0),
        max(0, 1 - metrics.get("typing_rhythm_variance", 0) / 1000.0),
        min(metrics.get("error_rate", 0) / 20.0, 1.0),
        metrics.get("mouse_movement_entropy", 0),
        min(metrics.get("mouse_click_rate", 0) / 60.0, 1.0),
        min(metrics.get("app_switch_frequency", 0) / 40.0, 1.0),
        min(metrics.get("focus_session_duration", 0) / 120.0, 1.0),
        min(metrics.get("idle_time", 0) / 600.0, 1.0),
        metrics.get("behavior_score", 50) / 100.0,
        metrics.get("hour_of_day", 12) / 24.0,
    ], dtype=np.float32)


# -----------------------------------------------------------
# Training Pipeline
# -----------------------------------------------------------
class BehavioralAnalyticsTrainer:
    """
    End-to-end training pipeline for behavioral analytics models.

    Step 1: Train Isolation Forest on normal behavior data
    Step 2: Train Autoencoder for embedding generation + anomaly detection
    Step 3: Export both models for inference
    """

    def __init__(
        self,
        model_dir: str = "/app/ml-models/behavioral-analytics",
        device: str = "cpu",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.scaler = StandardScaler()
        self.isolation_forest: IsolationForest | None = None
        self.autoencoder = BehavioralAutoencoder().to(self.device)

    def generate_synthetic_dataset(self, n_normal: int = 5000, n_anomalous: int = 500) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic behavioral dataset for initial training.
        Normal: typical office work patterns
        Anomalous: extreme fatigue patterns
        """
        rng = np.random.default_rng(42)

        # Normal behavior — typical ranges
        normal = np.column_stack([
            rng.normal(0.5, 0.1, n_normal),   # typing_speed ~50 WPM
            rng.normal(0.8, 0.1, n_normal),   # good rhythm
            rng.normal(0.05, 0.02, n_normal), # low error rate
            rng.normal(0.6, 0.15, n_normal),  # moderate mouse entropy
            rng.normal(0.3, 0.1, n_normal),   # moderate click rate
            rng.normal(0.2, 0.1, n_normal),   # low app switching
            rng.normal(0.5, 0.2, n_normal),   # moderate focus duration
            rng.normal(0.1, 0.05, n_normal),  # low idle time
            rng.normal(0.75, 0.1, n_normal),  # high behavior score
            rng.uniform(0.3, 0.9, n_normal),  # various work hours
        ])
        normal = np.clip(normal, 0, 1).astype(np.float32)

        # Anomalous — fatigue / high stress patterns
        anomalous = np.column_stack([
            rng.normal(0.2, 0.15, n_anomalous),  # very slow typing
            rng.normal(0.3, 0.2, n_anomalous),   # erratic rhythm
            rng.normal(0.4, 0.15, n_anomalous),  # high error rate
            rng.normal(0.2, 0.15, n_anomalous),  # low mouse entropy (sluggish)
            rng.normal(0.1, 0.08, n_anomalous),  # low click rate
            rng.normal(0.8, 0.15, n_anomalous),  # high app switching (distracted)
            rng.normal(0.1, 0.08, n_anomalous),  # very low focus
            rng.normal(0.7, 0.15, n_anomalous),  # high idle time
            rng.normal(0.2, 0.1, n_anomalous),   # low behavior score
            rng.uniform(0.0, 1.0, n_anomalous),  # any hour
        ])
        anomalous = np.clip(anomalous, 0, 1).astype(np.float32)

        return normal, anomalous

    def train_isolation_forest(self, X_normal: np.ndarray) -> None:
        """Train Isolation Forest on normal behavior samples."""
        logger.info("Training Isolation Forest", n_samples=len(X_normal))

        X_scaled = self.scaler.fit_transform(X_normal)

        self.isolation_forest = IsolationForest(
            n_estimators=200,
            max_samples="auto",
            contamination=0.05,
            max_features=1.0,
            random_state=42,
            n_jobs=-1,
        )
        self.isolation_forest.fit(X_scaled)
        logger.info("Isolation Forest trained")

    def train_autoencoder(
        self,
        X_normal: np.ndarray,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
    ) -> list[float]:
        """Train autoencoder on normal behavior data."""
        logger.info("Training Autoencoder", n_samples=len(X_normal), epochs=epochs)

        X_tensor = torch.from_numpy(X_normal).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

        optimizer = optim.AdamW(self.autoencoder.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.MSELoss()

        losses = []
        self.autoencoder.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch, in loader:
                optimizer.zero_grad()
                recon, _ = self.autoencoder(batch)
                loss = criterion(recon, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}", loss=f"{avg_loss:.6f}")

        self.autoencoder.eval()
        logger.info("Autoencoder training complete", final_loss=losses[-1])
        return losses

    def save_models(self) -> None:
        """Save trained models for deployment."""
        # Isolation Forest + scaler
        joblib.dump(self.isolation_forest, self.model_dir / "isolation_forest.pkl")
        joblib.dump(self.scaler, self.model_dir / "scaler.pkl")

        # Autoencoder weights
        torch.save(self.autoencoder.state_dict(), self.model_dir / "autoencoder.pt")

        # Export autoencoder to ONNX
        dummy_input = torch.randn(1, BehavioralAutoencoder.INPUT_DIM)
        torch.onnx.export(
            self.autoencoder,
            dummy_input,
            self.model_dir / "autoencoder.onnx",
            input_names=["features"],
            output_names=["reconstruction", "embedding"],
            dynamic_axes={"features": {0: "batch"}, "reconstruction": {0: "batch"}, "embedding": {0: "batch"}},
            opset_version=17,
        )

        logger.info("Models saved", dir=str(self.model_dir))

    def run_full_pipeline(self) -> None:
        """Execute the complete training pipeline."""
        X_normal, X_anomalous = self.generate_synthetic_dataset()

        self.train_isolation_forest(X_normal)
        self.train_autoencoder(X_normal)

        # Evaluate
        X_all = np.vstack([X_normal[:100], X_anomalous[:100]])
        y_true = np.array([1] * 100 + [-1] * 100)

        X_scaled = self.scaler.transform(X_all)
        if_preds = self.isolation_forest.predict(X_scaled)
        if_acc = (if_preds == y_true).mean()

        X_tensor = torch.from_numpy(X_all).to(self.device)
        ae_scores = self.autoencoder.get_anomaly_score(X_tensor).numpy()
        threshold = np.percentile(ae_scores[:100], 95)
        ae_preds = np.where(ae_scores > threshold, -1, 1)
        ae_acc = (ae_preds == y_true).mean()

        logger.info(
            "Evaluation complete",
            isolation_forest_accuracy=f"{if_acc:.2%}",
            autoencoder_accuracy=f"{ae_acc:.2%}",
        )

        self.save_models()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save models")
    args = parser.parse_args()
    trainer = BehavioralAnalyticsTrainer(model_dir=args.output_dir)
    trainer.run_full_pipeline()
