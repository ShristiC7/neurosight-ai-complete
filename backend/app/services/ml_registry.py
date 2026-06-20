"""
NeuroSight AI — ML Model Registry
Singleton registry that loads all models at startup and provides
thread-safe inference access throughout the application.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import structlog
import onnxruntime as ort
import numpy as np

from app.core.config import settings

logger = structlog.get_logger(__name__)


# -----------------------------------------------------------
# ONNX Inference Session Wrapper
# -----------------------------------------------------------
class ONNXModel:
    """
    Thread-safe ONNX Runtime inference wrapper.
    Used for production edge-optimized models.
    """

    def __init__(self, model_path: str, model_name: str) -> None:
        providers = self._get_providers()

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 4
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(
            model_path, sess_options=opts, providers=providers
        )
        self.name = model_name
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        logger.info(
            "ONNX model loaded",
            name=model_name,
            inputs=self.input_names,
            outputs=self.output_names,
            providers=providers,
        )

    def _get_providers(self) -> list[str]:
        available = ort.get_available_providers()
        if settings.INFERENCE_DEVICE == "cuda" and "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if settings.INFERENCE_DEVICE == "mps" and "CoreMLExecutionProvider" in available:
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference. Input keys must match self.input_names."""
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, inputs)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if elapsed_ms > settings.INFERENCE_TIMEOUT_MS:
            logger.warning(
                "Inference latency exceeded threshold",
                model=self.name,
                latency_ms=round(elapsed_ms, 2),
                threshold_ms=settings.INFERENCE_TIMEOUT_MS,
            )

        return dict(zip(self.output_names, outputs))


# -----------------------------------------------------------
# PyTorch Model Wrapper (for training/dev mode)
# -----------------------------------------------------------
def self_inference_mode(fn):
    """Decorator — ensures no gradients are computed."""
    import torch
    def wrapper(*args, **kwargs):
        with torch.inference_mode():
            return fn(*args, **kwargs)
    return wrapper

class TorchModel:
    """Wrapper for PyTorch models with lazy device placement."""

    def __init__(self, model: Any, model_name: str, device: str = "cpu") -> None:
        import torch
        self.model = model.to(device)
        self.model.eval()
        self.name = model_name
        self.device = device
        self._torch = torch

    @self_inference_mode
    def predict(self, **tensor_inputs) -> dict[str, Any]:
        """Run inference, returning Python scalars."""
        tensors = {
            k: v.to(self.device) if isinstance(v, self._torch.Tensor) else v
            for k, v in tensor_inputs.items()
        }
        with self._torch.inference_mode():
            return self.model(**tensors)


def self_inference_mode(fn):
    """Decorator — ensures no gradients are computed."""
    import torch
    def wrapper(*args, **kwargs):
        with torch.inference_mode():
            return fn(*args, **kwargs)
    return wrapper


# -----------------------------------------------------------
# Model Registry
# -----------------------------------------------------------
class _ModelRegistry:
    """
    Singleton registry for all NeuroSight AI models.
    Models are loaded once at startup and reused across requests.
    """

    def __init__(self) -> None:
        self._models: dict[str, ONNXModel | TorchModel] = {}
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """
        Load all models. Called once at application startup.
        Falls back to PyTorch models if ONNX files are not yet exported.
        """
        async with self._lock:
            if self._initialized:
                return

            logger.info("Initializing ML model registry...")

            await asyncio.gather(
                self._load_fatigue_model(),
                self._load_voice_stress_model(),
                self._load_behavioral_model(),
                self._load_productivity_model(),
                self._load_rl_agent(),
                return_exceptions=True,  # Don't fail startup if a model is missing
            )

            self._initialized = True
            logger.info("Model registry initialized", loaded=list(self._models.keys()))

    async def _load_fatigue_model(self) -> None:
        name = "eye_fatigue"
        onnx_path = Path(settings.EYE_FATIGUE_MODEL_PATH)

        if onnx_path.exists():
            self._models[name] = ONNXModel(str(onnx_path), name)
        else:
            logger.warning(
                "ONNX model not found, falling back to PyTorch",
                model=name,
                path=str(onnx_path),
            )
            try:
                from ml_models.eye_fatigue.src.model import create_fatigue_model
                model = create_fatigue_model(device=settings.INFERENCE_DEVICE)
                self._models[name] = TorchModel(model, name, device=settings.INFERENCE_DEVICE)
            except Exception as e:
                logger.error("Failed to load fatigue model", error=str(e))

    async def _load_voice_stress_model(self) -> None:
        name = "voice_stress"
        onnx_path = Path(settings.VOICE_STRESS_MODEL_PATH)

        if onnx_path.exists():
            self._models[name] = ONNXModel(str(onnx_path), name)
        else:
            logger.warning("Voice stress ONNX not found, skipping", model=name)

    async def _load_behavioral_model(self) -> None:
        """Load Isolation Forest for anomaly detection."""
        name = "behavioral_anomaly"
        model_path = Path(settings.BEHAVIORAL_MODEL_PATH)

        if model_path.exists():
            import joblib
            model = joblib.load(str(model_path))
            self._models[name] = model
            logger.info("Behavioral anomaly model loaded", model=name)
        else:
            logger.warning("Behavioral model not found, using rule-based fallback", model=name)

    async def _load_productivity_model(self) -> None:
        name = "productivity_predictor"
        model_path = Path(settings.PRODUCTIVITY_MODEL_PATH)

        if model_path.exists():
            self._models[name] = ONNXModel(str(model_path), name)
        else:
            logger.warning("Productivity model not found", model=name)

    async def _load_rl_agent(self) -> None:
        name = "rl_recommendation_agent"
        agent_path = Path(settings.RL_AGENT_PATH)

        if agent_path.exists():
            try:
                import sys
                sys.path.insert(0, str(Path(__file__).parents[3] / "ml-models"))
                from rl_agent.src.agent import ProductivityRLAgent
                agent = ProductivityRLAgent(device=settings.INFERENCE_DEVICE)
                agent.load(str(agent_path))
                self._models[name] = agent
                logger.info("RL agent loaded", model=name)
            except Exception as e:
                logger.error("Failed to load RL agent", error=str(e))
        else:
            logger.warning("RL agent checkpoint not found, using random policy", model=name)

    def get(self, name: str) -> ONNXModel | TorchModel | Any | None:
        """Retrieve a model by name. Returns None if not loaded."""
        return self._models.get(name)

    def loaded_models(self) -> list[str]:
        return list(self._models.keys())

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def get_fatigue_model(self) -> ONNXModel | TorchModel | None:
        return self.get("eye_fatigue")

    def get_voice_model(self) -> ONNXModel | TorchModel | None:
        return self.get("voice_stress")

    def get_behavioral_model(self):
        return self.get("behavioral_anomaly")

    def get_productivity_model(self) -> ONNXModel | TorchModel | None:
        return self.get("productivity_predictor")

    def get_rl_agent(self):
        return self.get("rl_recommendation_agent")


ModelRegistry = _ModelRegistry()
