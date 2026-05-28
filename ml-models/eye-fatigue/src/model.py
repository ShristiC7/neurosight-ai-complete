"""
NeuroSight AI — Eye Fatigue Detection Model
CNN + LSTM architecture for temporal drowsiness classification.

Architecture:
    Input: Eye crop images (48x48 grayscale) + EAR/MAR time series
    CNN branch: Feature extraction from eye images
    LSTM branch: Temporal modeling of EAR/MAR sequences
    Fusion: Concatenated features → classification head

Output: Drowsiness probability (5 classes: alert/mild/moderate/severe/critical)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# -----------------------------------------------------------
# CNN Branch — Eye Image Feature Extractor
# -----------------------------------------------------------
class EyeCNNEncoder(nn.Module):
    """
    Lightweight CNN for extracting features from 48x48 grayscale eye crops.
    Uses depthwise-separable convolutions for efficiency.
    """

    def __init__(self, output_dim: int = 128) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 48x48 → 24x24
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False),  # DW conv
            nn.Conv2d(32, 64, kernel_size=1, bias=False),  # PW conv
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            # Block 2: 24x24 → 12x12
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False),
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            # Block 3: 12x12 → 6x6
            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128, bias=False),
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(256, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, 1, 48, 48)
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.proj(x)
        return self.norm(x)


# -----------------------------------------------------------
# LSTM Branch — Temporal EAR/MAR Sequence Modeling
# -----------------------------------------------------------
class TemporalLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM for modeling EAR/MAR sequences over time.
    Captures drowsiness progression patterns.
    """

    def __init__(
        self,
        input_dim: int = 4,   # [EAR, MAR, blink_rate, head_tilt]
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, input_dim) — T is sequence length (e.g., 30 frames)
        x = self.input_proj(x)  # (B, T, hidden_dim)
        x = self.dropout(x)

        lstm_out, _ = self.lstm(x)  # (B, T, hidden_dim*2)

        # Attention pooling over time steps
        attn_weights = self.attention(lstm_out)  # (B, T, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (B, hidden_dim*2)

        return self.output_proj(context)


# -----------------------------------------------------------
# Fusion Head — Combine CNN + LSTM features
# -----------------------------------------------------------
class FatigueClassifier(nn.Module):
    """
    Main fatigue detection model.
    Fuses visual (CNN) and temporal (LSTM) features for classification.

    Classes:
        0: alert (EAR > 0.35, blink_rate 15-20/min)
        1: mild fatigue
        2: moderate fatigue
        3: severe fatigue
        4: critical / microsleep imminent
    """

    NUM_CLASSES = 5

    def __init__(
        self,
        cnn_dim: int = 128,
        lstm_dim: int = 128,
        fusion_dim: int = 256,
        dropout: float = 0.4,
        use_cnn: bool = True,
        use_lstm: bool = True,
    ) -> None:
        super().__init__()

        assert use_cnn or use_lstm, "At least one branch must be active"
        self.use_cnn = use_cnn
        self.use_lstm = use_lstm

        input_dim = 0
        if use_cnn:
            self.cnn_encoder = EyeCNNEncoder(output_dim=cnn_dim)
            input_dim += cnn_dim
        if use_lstm:
            self.lstm_encoder = TemporalLSTMEncoder(output_dim=lstm_dim)
            input_dim += lstm_dim

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(fusion_dim // 2, self.NUM_CLASSES),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward(
        self,
        eye_image: Tensor | None = None,
        temporal_features: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """
        Args:
            eye_image: (B, 1, 48, 48) - eye crop tensor
            temporal_features: (B, T, 4) - [EAR, MAR, blink_rate, head_tilt] sequence

        Returns:
            dict with 'logits', 'probabilities', 'fatigue_score'
        """
        features = []

        if self.use_cnn and eye_image is not None:
            cnn_feat = self.cnn_encoder(eye_image)
            features.append(cnn_feat)

        if self.use_lstm and temporal_features is not None:
            lstm_feat = self.lstm_encoder(temporal_features)
            features.append(lstm_feat)

        combined = torch.cat(features, dim=-1)
        logits = self.classifier(combined)
        probs = F.softmax(logits, dim=-1)

        # Continuous fatigue score (0-100)
        # Weighted sum: class 0→0, 1→25, 2→50, 3→75, 4→100
        weights = torch.tensor([0.0, 25.0, 50.0, 75.0, 100.0], device=probs.device)
        fatigue_score = (probs * weights).sum(dim=-1)

        return {
            "logits": logits,
            "probabilities": probs,
            "fatigue_score": fatigue_score,
            "predicted_class": probs.argmax(dim=-1),
        }

    @torch.inference_mode()
    def predict(
        self,
        eye_image: Tensor | None = None,
        temporal_features: Tensor | None = None,
    ) -> dict[str, float]:
        """Single-sample inference returning Python scalars."""
        self.eval()
        result = self.forward(eye_image, temporal_features)
        return {
            "fatigue_score": result["fatigue_score"].item(),
            "predicted_class": result["predicted_class"].item(),
            "probabilities": result["probabilities"].squeeze().tolist(),
        }


# -----------------------------------------------------------
# Training Utilities
# -----------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Critical fatigue events are rare → standard CE underweights them.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class FatigueDataAugmentation(nn.Module):
    """
    Data augmentation specific to eye/fatigue imagery.
    Applied during training to improve robustness.
    """

    def __init__(self, training: bool = True) -> None:
        super().__init__()
        self.training_mode = training

    def forward(self, x: Tensor) -> Tensor:
        if not self.training_mode:
            return x

        # Random horizontal flip (eye images are symmetric)
        if torch.rand(1) > 0.5:
            x = torch.flip(x, dims=[-1])

        # Random brightness jitter
        noise = torch.randn_like(x) * 0.05
        x = (x + noise).clamp(0, 1)

        # Random occlusion (simulates glasses, shadows)
        if torch.rand(1) > 0.7:
            h, w = x.shape[-2], x.shape[-1]
            oh = torch.randint(h // 4, h // 2, (1,)).item()
            ow = torch.randint(w // 4, w // 2, (1,)).item()
            top = torch.randint(0, h - oh, (1,)).item()
            left = torch.randint(0, w - ow, (1,)).item()
            x[..., top:top + oh, left:left + ow] = 0.0

        return x


# -----------------------------------------------------------
# Model Factory
# -----------------------------------------------------------
def create_fatigue_model(
    pretrained_path: str | None = None,
    device: str = "cpu",
) -> FatigueClassifier:
    """
    Factory function to instantiate and optionally load pretrained weights.
    """
    model = FatigueClassifier(
        cnn_dim=128,
        lstm_dim=128,
        fusion_dim=256,
        dropout=0.4,
    )

    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model
