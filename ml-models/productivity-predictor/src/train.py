"""
NeuroSight AI — Productivity Forecasting Training Pipeline
LSTM for temporal modeling + XGBoost for feature-based prediction.
Temporal Fusion Transformer (TFT) for advanced forecasting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import structlog

logger = structlog.get_logger(__name__)


# -----------------------------------------------------------
# LSTM Productivity Forecaster
# -----------------------------------------------------------
class ProductivityLSTM(nn.Module):
    """
    Sequence-to-scalar LSTM for productivity forecasting.

    Input: (B, T, feature_dim) — time-series of sensor readings
    Output: (B, 2) — [productivity_score, burnout_probability]

    Uses bidirectional LSTM + attention pooling for robust
    temporal pattern recognition.
    """

    INPUT_DIM = 15   # Matches FEATURE_NAMES in productivity_service.py
    HIDDEN_DIM = 256
    NUM_LAYERS = 3
    OUTPUT_DIM = 2   # productivity, burnout

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()

        self.input_norm = nn.LayerNorm(self.INPUT_DIM)

        self.lstm = nn.LSTM(
            input_size=self.INPUT_DIM,
            hidden_size=self.HIDDEN_DIM,
            num_layers=self.NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # Multi-head self-attention pooling
        self.attention = nn.MultiheadAttention(
            embed_dim=self.HIDDEN_DIM * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )

        # Regression heads
        self.productivity_head = nn.Sequential(
            nn.Linear(self.HIDDEN_DIM * 2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # 0-1 → scale to 0-100
        )

        self.burnout_head = nn.Sequential(
            nn.Linear(self.HIDDEN_DIM * 2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # 0-1 burnout probability
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: (B, T, INPUT_DIM)
        x = self.input_norm(x)
        lstm_out, _ = self.lstm(x)  # (B, T, HIDDEN_DIM*2)

        # Self-attention over time steps
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Global average pool + last hidden state
        pooled = attn_out.mean(dim=1)  # (B, HIDDEN_DIM*2)

        productivity = self.productivity_head(pooled).squeeze(-1) * 100  # 0-100
        burnout = self.burnout_head(pooled).squeeze(-1)                   # 0-1

        return {
            "productivity_score": productivity,
            "burnout_probability": burnout,
        }


# -----------------------------------------------------------
# Synthetic Dataset Generator
# -----------------------------------------------------------
class ProductivityDatasetGenerator:
    """
    Generates synthetic time-series productivity data for training.
    Simulates realistic work session patterns with known labels.
    """

    @staticmethod
    def generate(n_sessions: int = 2000, seq_len: int = 30) -> tuple:
        """Generate (X, y_productivity, y_burnout) arrays."""
        rng = np.random.default_rng(42)
        n_features = 15

        X = np.zeros((n_sessions, seq_len, n_features), dtype=np.float32)
        y_productivity = np.zeros(n_sessions, dtype=np.float32)
        y_burnout = np.zeros(n_sessions, dtype=np.float32)

        for i in range(n_sessions):
            # Simulate a work session with drift
            session_type = rng.choice(["peak", "normal", "declining", "exhausted"])

            if session_type == "peak":
                base_productivity = rng.uniform(0.75, 0.95)
                fatigue_drift = rng.uniform(0.0, 0.05)
                burnout_true = rng.uniform(0.0, 0.2)
            elif session_type == "normal":
                base_productivity = rng.uniform(0.5, 0.75)
                fatigue_drift = rng.uniform(0.02, 0.1)
                burnout_true = rng.uniform(0.1, 0.4)
            elif session_type == "declining":
                base_productivity = rng.uniform(0.3, 0.6)
                fatigue_drift = rng.uniform(0.05, 0.2)
                burnout_true = rng.uniform(0.4, 0.7)
            else:  # exhausted
                base_productivity = rng.uniform(0.1, 0.35)
                fatigue_drift = rng.uniform(0.15, 0.35)
                burnout_true = rng.uniform(0.7, 1.0)

            for t in range(seq_len):
                progress = t / seq_len
                fatigue = min(base_productivity - fatigue_drift * progress + rng.normal(0, 0.05), 1.0)

                X[i, t, 0] = np.clip(fatigue, 0, 1)       # fatigue_norm
                X[i, t, 1] = np.clip(burnout_true * progress + rng.normal(0, 0.05), 0, 1)  # stress
                X[i, t, 2] = np.clip(base_productivity - 0.1 * progress + rng.normal(0, 0.05), 0, 1)  # typing
                X[i, t, 3] = np.clip(rng.normal(0.5, 0.15), 0, 1)  # ear
                X[i, t, 4] = np.clip(base_productivity + rng.normal(0, 0.08), 0, 1)  # wpm
                X[i, t, 5] = rng.uniform(0, 0.5)  # rhythm_variance
                X[i, t, 6] = np.clip(fatigue_drift * progress + rng.normal(0, 0.02), 0, 1)  # errors
                X[i, t, 7] = rng.uniform(0.3, 0.8)  # mouse_entropy
                X[i, t, 8] = np.clip(burnout_true * 0.5 + rng.normal(0, 0.1), 0, 1)  # app_switch
                X[i, t, 9] = np.clip(base_productivity + rng.normal(0, 0.1), 0, 1)  # focus_duration
                X[i, t, 10] = progress  # session_duration
                hour = rng.uniform(8, 18)  # Work hours
                X[i, t, 11] = np.sin(2 * np.pi * hour / 24)
                X[i, t, 12] = np.cos(2 * np.pi * hour / 24)
                day = rng.integers(0, 6)  # Weekday (0-5)
                X[i, t, 13] = np.sin(2 * np.pi * day / 7)
                X[i, t, 14] = np.cos(2 * np.pi * day / 7)

            y_productivity[i] = np.clip(base_productivity * 100 - fatigue_drift * 30, 0, 100)
            y_burnout[i] = burnout_true

        return X, y_productivity, y_burnout


# -----------------------------------------------------------
# Training Pipeline
# -----------------------------------------------------------
class ProductivityTrainer:

    def __init__(self, model_dir: str = "/app/ml-models/productivity-predictor/src") -> None:
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_lstm(self, X: np.ndarray, y_prod: np.ndarray, y_burnout: np.ndarray) -> ProductivityLSTM:
        model = ProductivityLSTM().to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        # Use ceil to include any partial batch so the scheduler receives the correct total steps
        steps_per_epoch = (len(X) + 63) // 64  # equivalent to math.ceil(len(X) / 64)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, epochs=50, steps_per_epoch=steps_per_epoch
        )

        X_t = torch.from_numpy(X).to(self.device)
        y_p = torch.from_numpy(y_prod).to(self.device)
        y_b = torch.from_numpy(y_burnout).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_t, y_p, y_b)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        mse_loss = nn.MSELoss()
        bce_loss = nn.BCELoss()

        model.train()
        for epoch in range(50):
            total_loss = 0
            for Xb, yp, yb in loader:
                optimizer.zero_grad()
                out = model(Xb)
                loss = (
                    mse_loss(out["productivity_score"], yp) / 10000 +
                    bce_loss(out["burnout_probability"], yb)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                logger.info(f"LSTM Epoch {epoch+1}/50 — Loss: {total_loss/len(loader):.4f}")

        model.eval()
        return model

    def train_xgboost(self, X: np.ndarray, y_prod: np.ndarray) -> xgb.XGBRegressor:
        # Use last time step features for XGBoost (snapshot model)
        X_last = X[:, -1, :]
        X_train, X_val, y_train, y_val = train_test_split(X_last, y_prod, test_size=0.2, random_state=42)

        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=30,
            eval_metric="rmse",
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )

        val_pred = model.predict(X_val)
        rmse = mean_squared_error(y_val, val_pred) ** 0.5
        mae = mean_absolute_error(y_val, val_pred)
        logger.info(f"XGBoost — Val RMSE: {rmse:.2f}, MAE: {mae:.2f}")
        return model

    def run(self) -> None:
        logger.info("Generating synthetic dataset...")
        X, y_prod, y_burnout = ProductivityDatasetGenerator.generate(2000, 30)

        logger.info("Training LSTM forecaster...")
        lstm = self.train_lstm(X, y_prod, y_burnout)
        torch.save(lstm.state_dict(), self.model_dir / "lstm.pt")

        logger.info("Training XGBoost regressor...")
        xgb_model = self.train_xgboost(X, y_prod)
        xgb_model.save_model(self.model_dir / "xgboost.json")

        # Export LSTM to ONNX
        dummy = torch.randn(1, 30, 15)
        torch.onnx.export(
            lstm,
            dummy,
            self.model_dir / "lstm.onnx",
            input_names=["features"],
            output_names=["productivity_score", "burnout_probability"],
            dynamic_axes={"features": {0: "batch"}},
            opset_version=17,
        )

        logger.info("Productivity models saved!", dir=str(self.model_dir))


if __name__ == "__main__":
    ProductivityTrainer().run()
