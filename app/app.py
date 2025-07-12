from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from sr_model_top import upscale_image_from_path_or_url
from face_model_top import face_detection, face_detection_recover
from detr_model_top import detr_run
from yolo_model_top import yolo_detection


from PIL import Image

import io
camera = cv2.VideoCapture(0)  # or try (1) if external webcam

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"  # for session cookies
socketio = SocketIO(app, cors_allowed_origins="*")  # eventlet/gevent accelerate WS

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

    # Run face detection (saves crops to /Data internally)
    face_images = face_detection(img)
    # detr_images = detr_run(img)
    yolo_images = yolo_detection(img)

    print("face_images: ", len(face_images))
    print("yolo_images: ", len(yolo_images))
    # print("detr_images: ", len(detr_images))

    face_images = face_images + yolo_images

    print(face_images)

    sr_results = []
    
    for face_image in face_images:
        sr_face_img = upscale_image_from_path_or_url(face_image)
        buf = io.BytesIO()
        sr_face_img[0].save(buf, format='JPEG', quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode()
        data_uri = f'data:image/jpeg;base64,{b64}'
        sr_results.append(data_uri)

    return jsonify(sr=sr_results)
    
# @app.route('/recover', methods=['POST'])
# def recover():
#     face_images_r, o_w, o_h, o_pos_lists = face_detection_recover(img)
if __name__ == "__main__":
    # Use eventlet for production; disable Flask debug for speed.
    socketio.run(app, host="0.0.0.0", port=5000)
