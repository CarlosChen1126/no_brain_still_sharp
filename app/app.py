from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from sr_model_top import upscale_image_from_path_or_url
from face_model_top import face_detection

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

    face_images = face_detection(img)
    sr_face_imgs = []

    for face_image in face_images:
        sr_face_img = upscale_image_from_path_or_url(face_image)

        # Encode to base64
        buf = io.BytesIO()
        sr_face_img[0].save(buf, format='JPEG', quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode()
        data_uri = f'data:image/jpeg;base64,{b64}'
        sr_face_imgs.append(data_uri)  # ✅ collect all data URIs

    return jsonify(sr=sr_face_imgs)  # ✅ return the full list
    

if __name__ == "__main__":
    # Use eventlet for production; disable Flask debug for speed.
    socketio.run(app, host="0.0.0.0", port=5001)
