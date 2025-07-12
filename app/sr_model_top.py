from qai_hub_models.models._shared.super_resolution.app import SuperResolutionApp
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.base_model import TargetRuntime
from qai_hub_models.models.real_esrgan_general_x4v3.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    Real_ESRGAN_General_x4v3,
)
from PIL import Image
import threading

# Load the model once at import time
_model_instance = None
_model_lock = threading.Lock()

def _load_model_once():
    global _model_instance
    if _model_instance is None:
        with _model_lock:
            if _model_instance is None:
                print("📦 Loading ESRGAN model...")
                model_cls = Real_ESRGAN_General_x4v3
                parser = get_model_cli_parser(model_cls)
                parser = get_on_device_demo_parser(
                    parser,
                    add_output_dir=True,
                    available_target_runtimes=list(TargetRuntime.__members__.values()),
                )
                parser.add_argument("--image", type=str, default=str)
                args = parser.parse_args([])
                _model_instance = demo_model_from_cli_args(model_cls, MODEL_ID, args)
    return _model_instance

def upscale_image_from_path_or_url(image: Image.Image):
    """
    Takes a PIL.Image and returns the upscaled version using Qualcomm ESRGAN General x4v3 model.
    """
    model = _load_model_once()
    app = SuperResolutionApp(model)
    pred_images = app.upscale_image(image)
    return pred_images
