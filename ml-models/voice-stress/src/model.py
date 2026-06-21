"""
NeuroSight AI — Voice Stress Detection Model
CNN-based spectrogram classification + Transformer attention for temporal modeling.

Pipeline:
    Audio → MFCC/Spectrogram → CNN feature extractor → Transformer encoder
    → Multi-label emotion classification + stress score regression

Training datasets: RAVDESS, CREMA-D, custom annotated samples
Output: Emotion class probabilities + continuous stress score
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math


# -----------------------------------------------------------
# Spectrogram CNN Encoder
# -----------------------------------------------------------
class SpectrogramCNN(nn.Module):
    """
    2D CNN operating on mel-spectrograms.
    Treats spectrogram as (freq_bins, time_frames) image with 1 channel.

    Input: (B, 1, 128, T) — 128 mel bins, T time frames
    """

    def __init__(self, output_dim: int = 256) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            # Block 1: (1, 128, T) → (32, 64, T/2)
            self._conv_block(1, 32, kernel_size=(3, 3)),
            nn.MaxPool2d((2, 1)),  # Pool freq only, keep time

            # Block 2: → (64, 32, T/2)
            self._conv_block(32, 64, kernel_size=(3, 3)),
            nn.MaxPool2d((2, 1)),

            # Block 3: → (128, 16, T/2)
            self._conv_block(64, 128, kernel_size=(3, 3)),
            nn.MaxPool2d((2, 1)),

            # Block 4: → (256, 8, T/2)
            self._conv_block(128, 256, kernel_size=(3, 3)),
            nn.MaxPool2d((2, 1)),
        )

        # Collapse frequency dimension via mean (compatible with ONNX)
        self.freq_pool = None  # placeholder; not used
        self.proj = nn.Linear(256, output_dim)

    def _conv_block(self, in_ch: int, out_ch: int, kernel_size: tuple) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=(1, 1), groups=out_ch, bias=False),
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, 1, 128, T)
        x = self.encoder(x)          # (B, 256, freq', T')
        if self.freq_pool is not None:
            x = self.freq_pool(x)    # (B, 256, 1, T')
        else:
            # Collapse frequency dimension via mean (compatible with ONNX)
            x = x.mean(dim=2, keepdim=True)  # (B, 256, 1, T')
        x = x.squeeze(2).transpose(1, 2)  # (B, T', 256)
        x = self.proj(x)             # (B, T', output_dim)
        return x


# -----------------------------------------------------------
# Positional Encoding
# -----------------------------------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# -----------------------------------------------------------
# Transformer Encoder for Temporal Context
# -----------------------------------------------------------
class AudioTransformerEncoder(nn.Module):
    """
    Transformer encoder to capture long-range temporal dependencies in audio.
    Enables the model to detect stress patterns over time (not just instants).
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # CLS token for global representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, d_model)
        B = x.size(0)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, T+1, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)

        return x[:, 0, :]  # Return CLS token representation


# -----------------------------------------------------------
# MFCC Feature Branch
# -----------------------------------------------------------
class MFCCBranch(nn.Module):
    """
    Simple MLP branch for hand-crafted MFCC statistics.
    Complements CNN spectrogram features with engineered features.
    """

    def __init__(self, input_dim: int = 39, output_dim: int = 64) -> None:
        super().__init__()
        # 13 MFCC + 13 delta + 13 delta-delta = 39
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# -----------------------------------------------------------
# Voice Stress Model
# -----------------------------------------------------------
class VoiceStressModel(nn.Module):
    """
    Multimodal voice stress classifier.

    Inputs:
        spectrogram: (B, 1, 128, T) mel spectrogram
        mfcc_stats: (B, 39) MFCC + delta statistics

    Outputs:
        emotion_logits: (B, 5) — calm/stressed/fatigued/energetic/anxious
        stress_score: (B,) — continuous 0-100 stress score
    """

    EMOTION_CLASSES = ["calm", "stressed", "fatigued", "energetic", "anxious"]
    NUM_EMOTIONS = len(EMOTION_CLASSES)

    def __init__(
        self,
        cnn_dim: int = 256,
        transformer_layers: int = 4,
        mfcc_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.cnn = SpectrogramCNN(output_dim=cnn_dim)
        self.transformer = AudioTransformerEncoder(
            d_model=cnn_dim,
            nhead=8,
            num_layers=transformer_layers,
        )
        self.mfcc_branch = MFCCBranch(output_dim=mfcc_dim)

        fusion_dim = cnn_dim + mfcc_dim

        # Emotion classification head
        self.emotion_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, self.NUM_EMOTIONS),
        )

        # Stress score regression head (0-100)
        self.stress_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # 0-1 output, scaled to 0-100
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        spectrogram: Tensor,
        mfcc_stats: Tensor,
    ) -> dict[str, Tensor]:
        # CNN + Transformer on spectrogram
        cnn_feat = self.cnn(spectrogram)           # (B, T', cnn_dim)
        audio_repr = self.transformer(cnn_feat)    # (B, cnn_dim)

        # MFCC branch
        mfcc_feat = self.mfcc_branch(mfcc_stats)   # (B, mfcc_dim)

        # Fusion
        fused = torch.cat([audio_repr, mfcc_feat], dim=-1)  # (B, fusion_dim)

        # Heads
        emotion_logits = self.emotion_head(fused)
        stress_raw = self.stress_head(fused).squeeze(-1)
        stress_score = stress_raw * 100.0

        return {
            "emotion_logits": emotion_logits,
            "emotion_probs": F.softmax(emotion_logits, dim=-1),
            "emotion_class": emotion_logits.argmax(dim=-1),
            "stress_score": stress_score,
        }

    @torch.inference_mode()
    def predict(
        self,
        spectrogram: Tensor,
        mfcc_stats: Tensor,
    ) -> dict[str, float | str | list]:
        self.eval()
        output = self.forward(spectrogram, mfcc_stats)
        class_idx = output["emotion_class"].item()
        return {
            "stress_score": output["stress_score"].item(),
            "emotion_state": self.EMOTION_CLASSES[class_idx],
            "emotion_probs": {
                name: prob
                for name, prob in zip(
                    self.EMOTION_CLASSES,
                    output["emotion_probs"].squeeze().tolist(),
                )
            },
        }


def create_voice_stress_model(
    pretrained_path: str | None = None,
    device: str = "cpu",
) -> VoiceStressModel:
    model = VoiceStressModel(
        cnn_dim=256,
        transformer_layers=4,
        mfcc_dim=64,
        dropout=0.3,
    )
    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

    return model.to(device).eval()

# -----------------------------------------------------------
# Compatibility Stub
# -----------------------------------------------------------
class FatigueClassifier:
    """Stub class to satisfy imports expecting FatigueClassifier.

    The eye‑fatigue model provides the real implementation. This stub raises
    ``NotImplementedError`` when instantiated, ensuring that any accidental
    usage is caught during development while allowing the test suite to import
    the symbol without error.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "FatigueClassifier is not implemented in the voice‑stress module."
        )
