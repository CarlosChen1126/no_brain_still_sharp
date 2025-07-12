from qai_hub_models.models._shared.super_resolution.app import SuperResolutionApp
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
#from qai_hub_models.models.esrgan.model import ESRGAN, MODEL_ASSET_VERSION, MODEL_ID
from qai_hub_models.utils.base_model import BaseModel, TargetRuntime
#from qai_hub_models.models.real_esrgan_x4plus.model import (
    #MODEL_ASSET_VERSION,
    #MODEL_ID,
    #Real_ESRGAN_x4plus,
#)
from qai_hub_models.models.real_esrgan_general_x4v3.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    Real_ESRGAN_General_x4v3,
)

from PIL import Image

def upscale_image_from_path_or_url(image: Image.Image):
    """
    Takes a PIL.Image and returns the upscaled version using Qualcomm ESRGAN.
    """
    model_cls = Real_ESRGAN_x4plus
    model_id = MODEL_ID
    available_target_runtimes = list(TargetRuntime.__members__.values())

    parser = get_model_cli_parser(model_cls)
    parser = get_on_device_demo_parser(
        parser,
        add_output_dir=True,
        available_target_runtimes=available_target_runtimes,
    )

    parser.add_argument(
        "--image",
        type=str,
        default=str,
        help="image file path or URL.",
    )

    # Simulate CLI args
    is_test = True
    args = parser.parse_args([] if is_test else None)

    inference_model = demo_model_from_cli_args(
        model_cls,
        model_id,
        args,
    )

    app = SuperResolutionApp(inference_model)
    pred_images = app.upscale_image(image)

    return pred_images
