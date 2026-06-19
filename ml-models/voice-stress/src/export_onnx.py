import pathlib
import torch
import structlog
from model import create_voice_stress_model

logger = structlog.get_logger(__name__)

def export_onnx(output_dir: str = "./output"):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    # Load checkpoint
    checkpoint_path = pathlib.Path(output_dir) / "best_model.pt"
    if not checkpoint_path.is_file():
        logger.error("Checkpoint not found", path=str(checkpoint_path))
        return
    model = create_voice_stress_model(device="cpu")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    dummy_spec = torch.randn(1, 1, 128, 100)
    dummy_mfcc = torch.randn(1, 39)
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
    logger.info("ONNX model exported", path=str(onnx_path))

if __name__ == "__main__":
    export_onnx()
