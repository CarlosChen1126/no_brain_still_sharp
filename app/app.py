from __future__ import annotations

import base64
import io
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from PIL import Image

from sr_model_top import upscale_image_from_path_or_url
from face_model_top import face_detection, face_detection_recover, pic_recover
from detr_model_top import detr_run

# Initialize Flask and SocketIO
app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize camera if needed
camera = cv2.VideoCapture(0)

@app.route("/")
def index():
    """Serve the main HTML page."""
    return render_template("./index.html")

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if file is None:
        return jsonify(error='No file uploaded'), 400

    try:
        img = Image.open(file.stream).convert('RGB')
    except Exception as exc:
        return jsonify(error=f'Bad image: {exc}'), 400

    # Run detection
    face_images = face_detection(img)  # Returns list of face images
    detr_images, detr_tags = detr_run(img)        # Returns list of (Image, label) tuples or just Images
    print("detr_tags: ", detr_tags)

    # # Normalize detr_images to be (Image, label)
    # formatted_detr_images = []
    # for item in detr_images:
    #     if isinstance(item, tuple):
    #         formatted_detr_images.append(item)
    #     else:
    #         formatted_detr_images.append((item, "unknown"))
    formatted_detr_images = list(zip(detr_images, detr_tags))


    # Combine all detections
    all_detections = [(im, "people") for im in face_images] + formatted_detr_images

    sr_results = []
    for cropped_img, label in all_detections:
        sr_face_img = upscale_image_from_path_or_url(cropped_img)
        if not sr_face_img:
            continue

        buf = io.BytesIO()
        sr_face_img[0].save(buf, format='JPEG', quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode()
        data_uri = f'data:image/jpeg;base64,{b64}'

        sr_results.append({
            "src": data_uri,
            "type": label
        })

    return jsonify(sr=sr_results)

@app.route('/recover', methods=['POST'])
def recover():
    file = request.files.get('file')
    if file is None:
        return jsonify(error='No file uploaded'), 400

    try:
        img = Image.open(file.stream).convert('RGB')
    except Exception as exc:
        return jsonify(error=f'Bad image: {exc}'), 400

    face_images_r, o_w, o_h, o_pos_lists = face_detection_recover(img)
    print("o pos lists: ", o_pos_lists)
    sr_results = []
    for face_image in face_images_r:
        sr_face_img = upscale_image_from_path_or_url(face_image)
        sr_results.append(sr_face_img[0])

    rec_pic = pic_recover(sr_results, o_w, o_h, o_pos_lists, img)
    buf = io.BytesIO()
    rec_pic.save(buf, format='JPEG', quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    data_uri = f'data:image/jpeg;base64,{b64}'

    return jsonify(sr=data_uri)

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5001)
