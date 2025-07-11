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

from qai_hub_models.models._shared.detr.coco_label_map import LABEL_MAP
import cv2

# Run DETR app end-to-end on a sample image.
# The demo will display the predicted mask in a window.
def detr_demo(
    model_cls: type[BaseModel],
    model_id: str,
    default_weights: str,
    default_image: str | CachedWebAsset,
    is_test: bool = False,
):
    # Demo parameters
    parser = get_model_cli_parser(model_cls)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=default_image,
        help="test image file path or URL",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, model_id)

    # Load image & model
    detr = demo_model_from_cli_args(model_cls, model_id, args)
    if isinstance(detr, model_cls):
        input_spec = detr.get_input_spec()
    else:
        input_spec = model_cls.get_input_spec()
    (h, w) = input_spec["image"][0][2:]

    cap = cv2.VideoCapture(0)
    app = DETRApp(detr, h, w)
    while True:
        # 讀取一幀畫面
        ret, frame = cap.read()
        if ret:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pred_images, _, labels, _ = app.predict(pil_image, default_weights, 0.5)
            pred_image = Image.fromarray(pred_images[0])
            if labels.numel() != 0:
                for label in labels:
                    print(LABEL_MAP[label.item()])
        if not ret:
            print("無法讀取影像")
            break

        # 顯示畫面
        cv2.imshow('Camera Feed', frame)

        # 每 1 毫秒檢查是否按下 'q' 鍵
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("結束影像輸出")
            break
    # Run app to scores, labels and boxes
    # img = load_image(args.image)

    cap.release()
    cv2.destroyAllWindows()
    # Show the predicted boxes, scores and class names on the image.
    if is_test:
        assert isinstance(pred_image, Image.Image)
    else:
        display_or_save_image(pred_image, args.output_dir)
