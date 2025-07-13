from qai_hub_models.models._shared.super_resolution.app import SuperResolutionApp
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
# from qai_hub_models.models.xlsr.model import MODEL_ID, XLSR
# from qai_hub_models.models.esrgan.model import ESRGAN, MODEL_ASSET_VERSION, MODEL_ID
from qai_hub_models.utils.base_model import BaseModel, TargetRuntime
# from qai_hub_models.models.real_esrgan_x4plus.model import (
#     MODEL_ASSET_VERSION,
#     MODEL_ID,
#     Real_ESRGAN_x4plus,
# )
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

def load_sr_model():
    model_cls = Real_ESRGAN_General_x4v3
    parser = get_model_cli_parser(model_cls)
    parser = get_on_device_demo_parser(
        parser,
        add_output_dir=True,
        available_target_runtimes=list(TargetRuntime.__members__.values()),
    )
    parser.add_argument("--image", type=str, default=str)
    args = parser.parse_args([])
    model = demo_model_from_cli_args(model_cls, MODEL_ID, args)
    return model

def upscale_image_from_path_or_url(model, image: Image.Image):
    """
    Takes a PIL.Image and returns the upscaled version using Qualcomm ESRGAN General x4v3 model.
    """
    # model = _load_model_once()
    # app = SuperResolutionApp(model)
    # model_cls = Real_ESRGAN_General_x4v3
    # model_id = MODEL_ID
    # available_target_runtimes = list(TargetRuntime.__members__.values())

    # parser = get_model_cli_parser(model_cls)
    # parser = get_on_device_demo_parser(
    #     parser,
    #     add_output_dir=True,
    #     available_target_runtimes=available_target_runtimes,
    # )

    # parser.add_argument(
    #     "--image",
    #     type=str,
    #     default=str,
    #     help="image file path or URL.",
    # )

    # # Simulate CLI args
    # is_test = True
    # args = parser.parse_args([] if is_test else None)

    # inference_model = demo_model_from_cli_args(
    #     model_cls,
    #     model_id,
    #     args,
    # )

    app = SuperResolutionApp(model)
    pred_images = app.upscale_image(image)
    return pred_images
