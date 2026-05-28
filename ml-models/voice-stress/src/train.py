"""
Voice Stress Model – Training Script
Generates a synthetic dataset of spectrograms and MFCC statistics, trains the model, saves the best checkpoint, and exports to ONNX.
"""

import pathlib
import structlog
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from model import create_voice_stress_model, VoiceStressModel

logger = structlog.get_logger(__name__)

# -----------------------------------------------------------
# Synthetic data generator
# -----------------------------------------------------------
class SyntheticVoiceDataset:
    """Generate random spectrograms and MFCC stats.
    Spectrogram shape: (1, 128, T) – we use T=100.
    MFCC stats shape: (39,).
    Labels:
      - emotion (0‑4) random class
      - stress score (0‑100) random float
    """

    def __init__(self, n_samples: int = 2000, time_frames: int = 100):
        self.n_samples = n_samples
        self.time_frames = time_frames
        rng = torch.Generator().manual_seed(42)
        # Spectrograms: (N, 1, 128, T)
        self.spectrograms = torch.randn(n_samples, 1, 128, time_frames, generator=rng)
        # MFCC stats: (N, 39)
        self.mfcc_stats = torch.randn(n_samples, 39, generator=rng)
        self.emotion_labels = torch.randint(0, 5, (n_samples,), generator=rng)
        self.stress_scores = torch.rand(n_samples, generator=rng) * 100.0

    def as_tensors(self):
        return (
            self.spectrograms,
            self.mfcc_stats,
            self.emotion_labels,
            self.stress_scores,
        )

# -----------------------------------------------------------
# Training utilities
# -----------------------------------------------------------
def train_one_epoch(model: VoiceStressModel, loader: DataLoader, optimizer, device):
    model.train()
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    total_loss = 0.0
    for spect, mfcc, emo, stress in loader:
        spect, mfcc, emo, stress = (
            spect.to(device),
            mfcc.to(device),
            emo.to(device),
            stress.to(device),
        )
        optimizer.zero_grad()
        out = model(spect, mfcc)
        loss = ce_loss(out["emotion_logits"], emo) + mse_loss(out["stress_score"], stress)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model: VoiceStressModel, loader: DataLoader, device):
    model.eval()
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    total_loss = 0.0
    with torch.inference_mode():
        for spect, mfcc, emo, stress in loader:
            spect, mfcc, emo, stress = (
                spect.to(device),
                mfcc.to(device),
                emo.to(device),
                stress.to(device),
            )
            out = model(spect, mfcc)
            loss = ce_loss(out["emotion_logits"], emo) + mse_loss(out["stress_score"], stress)
            total_loss += loss.item()
    return total_loss / len(loader)

# -----------------------------------------------------------
# Main training loop
# -----------------------------------------------------------
def main(
    output_dir: str = "./output",
    epochs: int = 30,
    batch_size: int = 64,
    device: str = "cpu",
):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Generating synthetic voice‑stress dataset…")
    dataset = SyntheticVoiceDataset()
    X_spec, X_mfcc, y_emo, y_stress = dataset.as_tensors()
    full_dataset = TensorDataset(X_spec, X_mfcc, y_emo, y_stress)
    train_len = int(0.8 * len(full_dataset))
    val_len = len(full_dataset) - train_len
    train_set, val_set = random_split(full_dataset, [train_len, val_len])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = create_voice_stress_model(device=device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, epochs=epochs, steps_per_epoch=len(train_loader)
    )

    best_val = float("inf")
    best_path = pathlib.Path(output_dir) / "best_model.pt"
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step()
        logger.info(
            f"Epoch {epoch}/{epochs} — train loss: {train_loss:.4f}, val loss: {val_loss:.4f}"
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            logger.info("Saved new best checkpoint", path=str(best_path))

    # Export to ONNX (batch size 1 for safety)
    dummy_spec = torch.randn(1, 1, 128, 100, device=device)
    dummy_mfcc = torch.randn(1, 39, device=device)
    onnx_path = pathlib.Path(output_dir) / "voice_stress.onnx"
    torch.onnx.export(
        model,
        (dummy_spec, dummy_mfcc),
        onnx_path,
        input_names=["spectrogram", "mfcc_stats"],
        output_names=["emotion_logits", "stress_score"],
        dynamic_axes={"spectrogram": {2: "time"}},
        opset_version=17,
    )
    logger.info("Voice‑stress model exported", dir=str(onnx_path.parent))

if __name__ == "__main__":
    main()
