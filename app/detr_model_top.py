# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from PIL import Image

from qai_hub_models.models._shared.detr.app import DETRApp
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebAsset, load_image
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.display import display_or_save_image
from qai_hub_models.models.conditional_detr_resnet50.model import (
    DEFAULT_WEIGHTS,
    MODEL_ID,
    ConditionalDETRResNet50,
)
from gsr_model_top import upscale_image_from_path_or_url  # ✅ Import your SR function

# Run DETR app end-to-end on a sample image.
# The demo will display the predicted mask in a window.
# def detr_run(
#     model_cls: type[BaseModel],
#     model_id: str,
#     default_weights: str,
#     default_image: str | CachedWebAsset,
#     is_test: bool = False,
# ):
from typing import List, Tuple

def map_boxes_back_to_original(
    boxes: List[Tuple[float, float, float, float]],
    original_size: Tuple[int, int],
    resized_size: Tuple[int, int]
) -> List[Tuple[int, int, int, int]]:
    """
    Map bounding boxes from resized image back to original image dimensions.

    Args:
        boxes: List of bounding boxes [(x1, y1, x2, y2)] in resized image.
        original_size: (width, height) of the original image.
        resized_size: (width, height) of the resized image.

    Returns:
        List of boxes mapped back to original image size.
    """
    orig_w, orig_h = original_size
    resized_w, resized_h = resized_size

    scale_x = orig_w / resized_w
    scale_y = orig_h / resized_h

    mapped_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        mapped_box = (
            int(x1 * scale_x),
            int(y1 * scale_y),
            int(x2 * scale_x),
            int(y2 * scale_y)
        )
        mapped_boxes.append(mapped_box)

    return mapped_boxes


def extract_bounding_boxes(img: Image.Image, boxes: list):
    crops = []
    for box in boxes:
        x, y, rx, ry = box
        crop = img.crop((int(x), int(y), int(rx), int(ry)))
        crops.append(crop)
    return crops

def resize_to_multiple_of_32(img):
    w, h = img.size
    return img.resize(((w + 31) // 32 * 32, (h + 31) // 32 * 32))


def detr_run(img: Image.Image):
    # Demo parameters
    # parser = get_model_cli_parser(model_cls)
    parser = get_model_cli_parser(ConditionalDETRResNet50)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    # parser.add_argument(
    #     "--image",
    #     type=str,
    #     default=default_image,
    #     help="test image file path or URL",
    # )
    # args = parser.parse_args([] if is_test else None)
    args = parser.parse_args([])
    validate_on_device_demo_args(args, MODEL_ID)

    # Load image & model
    detr = demo_model_from_cli_args(ConditionalDETRResNet50, MODEL_ID, args)
    if isinstance(detr, ConditionalDETRResNet50):
        input_spec = detr.get_input_spec()
    else:
        input_spec = ConditionalDETRResNet50.get_input_spec()
    # (h, w) = input_spec["image"][0][2:]
    # w, h = image.size

    # Run app to scores, labels and boxes
    # img = load_image(args.image)
    w, h = img.size  # (w, h) of original image
    # resized_image = resize_to_multiple_of_32(img)
    # resized_size = resized_image.size
    app = DETRApp(detr, h, w)
    pred_images, scores, labels, boxes = app.predict(img, DEFAULT_WEIGHTS, threshold = 0.7)
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)  # x1, x2
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)  # y1, y2
    crops = extract_bounding_boxes(img, boxes)
    # pred_image = Image.fromarray(pred_images[0])

    tags_tmp = []
    for label in labels:
        if label >= 2 and label <= 9: # vehicles
            tags_tmp.append("vehicles")
        elif label >= 16 and label <= 25: # animals
            tags_tmp.append("animals")
        else:
            tags_tmp.append("none")

    enhanced_crops = []
    tags = []
    for i, crop in enumerate(crops):
        if tags_tmp[i] != "none":
            tags.append(tags_tmp[i])
            sr = upscale_image_from_path_or_url(crop)
            if sr:
                enhanced_crops.append(sr[0])
            else:
                enhanced_crops.append(crop)

    # is_test = False

    # Show the predicted boxes, scores and class names on the image.
    # if is_test:
    #     assert isinstance(pred_image, Image.Image)
    # else:
    #     display_or_save_image(pred_image, args.output_dir)

    print("------done------")
    print("h: ", h)
    print("w: ", w)
    print("scores: ", scores)
    print("labels: ", labels)
    print("boxes: ", boxes)
    print("------done------")

    return enhanced_crops, tags