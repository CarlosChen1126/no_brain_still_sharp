import os
import ast
from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image

from typing import List

from sr_model_top import upscale_image_from_path_or_url  # ✅ Import your SR function
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

_last_save_time = datetime.min
SAVE_INTERVAL = timedelta(seconds=6)

def extract_bounding_boxes(img: Image.Image, boxes: list, top_pad_ratio=0.4, side_pad_ratio=0.1):
    crops = []
    img_w, img_h = img.size

    for box in boxes:
        x, y, w, h, _ = box

        # Expand the bounding box
        pad_top = int(h * top_pad_ratio)
        pad_side = int(w * side_pad_ratio)

        left = max(int(x - pad_side), 0)
        upper = max(int(y - pad_top), 0)
        right = min(int(x + w + pad_side), img_w)
        lower = min(int(y + h + pad_side), img_h)

        crop = img.crop((left, upper, right, lower))
        crops.append(crop)
    return crops

def extract_bounding_boxes_recover(img: Image.Image, boxes: list):
    crops = []
    pos_lists = []
    for box in boxes:
        pos_list = []
        x, y, w, h, _ = box
        
        left = int(x)
        upper = int(y)
        right = int(x + w)
        lower = int(y + h)
        
        crop = img.crop((left, upper, right, lower))
        pos_list.append(x)
        pos_list.append(y)
        pos_list.append(w)
        pos_list.append(h)
        pos_lists.append(pos_list)
        crops.append(crop)
    return crops, pos_lists
def resize_to_multiple_of_32(img):
    w, h = img.size
    return img.resize(((w + 31) // 32 * 32, (h + 31) // 32 * 32))

def manage_folders(base_dir: Path, max_folders=5):
    folders = sorted([f for f in base_dir.iterdir() if f.is_dir()])
    while len(folders) > max_folders:
        oldest = folders.pop(0)
        for file in oldest.iterdir():
            file.unlink()
        oldest.rmdir()

def should_save_image():
    global _last_save_time
    now = datetime.now()
    if now - _last_save_time >= SAVE_INTERVAL:
        _last_save_time = now
        return True
    return False

def resize_to_multiple_of_32_recover(img):
    w, h = img.size
    new_w = (w + 31) // 32 * 32
    new_h = (h + 31) // 32 * 32
    resized_img = img.resize((new_w, new_h))
    return resized_img, w, h

def load_face_model():
    parser = get_model_cli_parser(FaceDetLite)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument("--image", type=str, default=str)
    args = parser.parse_args([])
    model = demo_model_from_cli_args(FaceDetLite, MODEL_ID, args)
    return model

def face_detection(model, image: Image.Image):
    # validate_on_device_demo_args(args, MODEL_ID)
    print("old image size: ", image.size)
    resized_image = resize_to_multiple_of_32(image)
    print("resized image size: ", resized_image.size)
    print("Model Loaded")

    app = FaceDetLiteApp(model)
    res, _ = app.run_inference_on_image(resized_image)
    boxes = ast.literal_eval(str(res))
    crops = extract_bounding_boxes(resized_image, boxes)
    # print("small image size: ", crops[0].size)
    print(boxes)

    enhanced_crops = []
    for crop in crops:
        # sr = upscale_image_from_path_or_url(crop)
        # print("small optimized image size: ", sr[0].size)
        # if sr:
        #     enhanced_crops.append(sr[0])
        # else:
        enhanced_crops.append(crop)

    # Save logic every 6 seconds only
    if should_save_image():
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        save_dir = Path("Data") / timestamp
        save_dir.mkdir(parents=True, exist_ok=True)

        existing_files = sorted(save_dir.glob("*.jpg"))
        remaining_slots = 10 - len(existing_files)

        for i, crop in enumerate(enhanced_crops[:remaining_slots]):
            filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}.jpg"
            crop.save(save_dir / filename)

        manage_folders(Path("Data"))

    print("enhanced_crops: ", enhanced_crops)

    return enhanced_crops
    # for i, crop in enumerate(crops):
    #     face_images.append(crop)
    #     # crop.save(f"crop_{i}.jpg")
    # return face_images


def face_detection_recover(image: Image.Image):
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
    resized_image, original_w, original_h = resize_to_multiple_of_32_recover(image)
    print("Model Loaded")

    app = FaceDetLiteApp(model)
    res, out = app.run_inference_on_image(resized_image)
    out_dict = {}
    face_images = []
    out_dict["bounding box"] = str(res)
    boxes = ast.literal_eval(out_dict["bounding box"])
    crops, original_pos_lists = extract_bounding_boxes_recover(resized_image, boxes)
    original_pos_lists = rescale_boxes(original_pos_lists, resized_image.size, (original_w, original_h))

    for i, crop in enumerate(crops):
        face_images.append(crop)
        # crop.save(f"crop_{i}.jpg")
    return face_images, original_w, original_h, original_pos_lists
def rescale_boxes(boxes, resized_size, original_size):
    resized_w, resized_h = resized_size
    original_w, original_h = original_size
    scale_x = original_w / resized_w
    scale_y = original_h / resized_h

    scaled_boxes = []
    for x, y, w, h in boxes:
        scaled_x = x * scale_x
        scaled_y = y * scale_y
        scaled_w = w * scale_x
        scaled_h = h * scale_y
        scaled_boxes.append([scaled_x, scaled_y, scaled_w, scaled_h])
    return scaled_boxes
def paste_boxes_back_to_image(img: Image.Image, box: list, small_img: Image.Image) -> Image.Image:
    img_copy = img.copy()

    x, y, w, h = box
    left = int(x)
    upper = int(y)
    right = int(x + w)
    lower = int(y + h)
    # print("[back] small image size: ", small_img.size)
    # print(box)
    resized_small = small_img.resize((int(w), int(h)))
    # print("[back] resized small image size: ", resized_small.size)
    img_copy.paste(resized_small, (left, upper))

    return img_copy
def pic_recover(images: List[Image.Image], original_w, original_h, original_pos_lists, o_img):
    for i, image in enumerate(images):
        o_img = paste_boxes_back_to_image(o_img, original_pos_lists[i], image)
    return o_img
