"""
NeuroSight AI — Behavioral Analytics Service
Isolation Forest + Autoencoder anomaly detection on behavioral features.
"""
from pathlib import Path
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class BehavioralService:
    """
    Wraps the trained Isolation Forest and Autoencoder models.
    Falls back to a heuristic score if models are not trained yet.
    """
    _iso_forest = None
    _scaler = None
    _autoencoder = None
    _initialized = False

    @classmethod
    def _load_models(cls) -> None:
        if cls._initialized:
            return
        model_dir = Path("/app/ml-models/behavioral-analytics")
        try:
            import joblib
            cls._scaler = joblib.load(model_dir / "scaler.pkl")
            cls._iso_forest = joblib.load(model_dir / "isolation_forest.pkl")
            logger.info("Behavioral Isolation Forest loaded")
        except Exception as e:
            logger.warning("Behavioral IF model not loaded", error=str(e))

        try:
            import onnxruntime as ort
            cls._autoencoder = ort.InferenceSession(
                str(model_dir / "autoencoder.onnx"),
                providers=["CPUExecutionProvider"],
            )
            logger.info("Behavioral Autoencoder loaded (ONNX)")
        except Exception as e:
            logger.warning("Behavioral autoencoder not loaded", error=str(e))

        cls._initialized = True

    def compute_anomaly_score(self, metrics: dict) -> float:
        """
        Returns anomaly score in [0, 1].
        0 = completely normal, 1 = highly anomalous.
        """
        self._load_models()
        features = self._build_features(metrics)

        if self._iso_forest is not None and self._scaler is not None:
            try:
                X = self._scaler.transform(features.reshape(1, -1))
                # Isolation Forest: -1=anomaly, 1=normal. Score: lower = more anomalous
                score = self._iso_forest.decision_function(X)[0]
                # Map from [-0.5, 0.5] range to [0, 1]
                return float(np.clip(0.5 - score, 0, 1))
            except Exception as e:
                logger.warning("IF inference failed", error=str(e))

        # Heuristic fallback
        return self._heuristic_anomaly(metrics)

    def get_embedding(self, metrics: dict) -> list[float]:
        """Generate 256-dim behavioral embedding for Qdrant storage."""
        self._load_models()
        features = self._build_features(metrics)
        if self._autoencoder is not None:
            try:
                outputs = self._autoencoder.run(
                    None, {"features": features.reshape(1, -1).astype(np.float32)}
                )
                # outputs[1] is the embedding
                return outputs[1].flatten().tolist()
            except Exception as e:
                logger.warning("Autoencoder embedding failed", error=str(e))
        return [0.0] * 256

    @staticmethod
    def _build_features(m: dict) -> np.ndarray:
        return np.array([
            min(m.get("typing_speed", 0) / 100.0, 1.0),
            max(0, 1 - m.get("typing_rhythm_variance", 0) / 1000.0),
            min(m.get("error_rate", 0) / 20.0, 1.0),
            m.get("mouse_movement_entropy", 0),
            min(m.get("mouse_click_rate", 0) / 60.0, 1.0),
            min(m.get("app_switch_frequency", 0) / 40.0, 1.0),
            min(m.get("focus_session_duration", 0) / 120.0, 1.0),
            min(m.get("idle_time", 0) / 600.0, 1.0),
            m.get("behavior_score", 50) / 100.0,
            m.get("hour_of_day", 12) / 24.0,
        ], dtype=np.float32)

    @staticmethod
    def _heuristic_anomaly(m: dict) -> float:
        """Simple rule-based anomaly scoring when model is unavailable."""
        score = 0.0
        if m.get("typing_speed", 50) < 15:        score += 0.3
        if m.get("error_rate", 2) > 8:            score += 0.25
        if m.get("app_switch_frequency", 5) > 30: score += 0.2
        if m.get("focus_session_duration", 30) < 3: score += 0.25
        return float(np.clip(score, 0, 1))
