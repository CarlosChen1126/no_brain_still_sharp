from PIL import Image
import ast

from qai_hub_models.models.face_det_lite.app import FaceDetLiteApp
from qai_hub_models.models.face_det_lite.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    FaceDetLite,
)
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)

def extract_bounding_boxes(img: Image.Image, boxes: list):
    crops = []
    for box in boxes:
        x, y, w, h, _ = box
        
        left = int(x)
        upper = int(y)
        right = int(x + w)
        lower = int(y + h)
        
        crop = img.crop((left, upper, right, lower))
        crops.append(crop)
    return crops
def resize_to_multiple_of_32(img):
    w, h = img.size
    new_w = (w + 31) // 32 * 32
    new_h = (h + 31) // 32 * 32
    resized_img = img.resize((new_w, new_h))
    return resized_img

def face_detection(image: Image.Image):
    parser = get_model_cli_parser(FaceDetLite)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=str,
        help="image file path or URL",
    )
    is_test = False
    args = parser.parse_args([] if is_test else None)
    model = demo_model_from_cli_args(FaceDetLite, MODEL_ID, args)
    validate_on_device_demo_args(args, MODEL_ID)
    resized_image = resize_to_multiple_of_32(image)
    print("Model Loaded")

    app = FaceDetLiteApp(model)
    res, out = app.run_inference_on_image(resized_image)
    out_dict = {}
    face_images = []
    out_dict["bounding box"] = str(res)
    boxes = ast.literal_eval(out_dict["bounding box"])
    crops = extract_bounding_boxes(resized_image, boxes)
    for i, crop in enumerate(crops):
        face_images.append(crop)
        crop.save(f"crop_{i}.jpg")
    return face_images
