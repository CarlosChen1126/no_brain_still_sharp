from qai_hub_models.models._shared.super_resolution.app import SuperResolutionApp
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
# from qai_hub_models.models.xlsr.model import MODEL_ID, XLSR
from qai_hub_models.models.esrgan.model import ESRGAN, MODEL_ASSET_VERSION, MODEL_ID
from qai_hub_models.utils.base_model import BaseModel, TargetRuntime
from PIL import Image

def upscale_image_from_path_or_url(image: Image.Image):
    model_cls = ESRGAN
    model_id = MODEL_ID
    available_target_runtimes = list(TargetRuntime.__members__.values())

    print(model_id)
    print(model_cls.from_pretrained) 
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