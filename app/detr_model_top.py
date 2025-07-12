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
from sr_model_top import upscale_image_from_path_or_url  # ✅ Import your SR function

# Run DETR app end-to-end on a sample image.
# The demo will display the predicted mask in a window.
# def detr_run(
#     model_cls: type[BaseModel],
#     model_id: str,
#     default_weights: str,
#     default_image: str | CachedWebAsset,
#     is_test: bool = False,
# ):


def extract_bounding_boxes(img: Image.Image, boxes: list):
    crops = []
    for box in boxes:
        x, y, w, h = box
        crop = img.crop((int(x), int(y), int(x + w), int(y + h)))
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
    (h, w) = input_spec["image"][0][2:]

    # Run app to scores, labels and boxes
    # img = load_image(args.image)
    resized_image = resize_to_multiple_of_32(img)
    app = DETRApp(detr, h, w)
    pred_images, scores, labels, boxes = app.predict(resized_image, DEFAULT_WEIGHTS, threshold = 0.5)
    crops = extract_bounding_boxes(resized_image, boxes)
    # pred_image = Image.fromarray(pred_images[0])

    enhanced_crops = []
    for crop in crops:
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
    print("scores: ", scores)
    print("labels: ", labels)
    print("boxes: ", boxes)
    print("------done------")

    return enhanced_crops