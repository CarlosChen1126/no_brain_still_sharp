from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from model_top import upscale_image_from_path_or_url

from PIL import Image

import io

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"  # for session cookies
socketio = SocketIO(app, cors_allowed_origins="*")  # eventlet/gevent accelerate WS

@app.route("/")
def index():
    """Serve the main HTML page."""
    return render_template("./index.html")

@app.route('/upload', methods=['POST'])
def upload():
    """
    Browser sends a FormData with a field called “file”.
    We read it, super-resolve it, return { sr: data-URI }.
    """
    file = request.files.get('file')
    if file is None:
        return jsonify(error='No file uploaded'), 400

    try:
        # Read webcam snapshot -> PIL
        img = Image.open(file.stream).convert('RGB')
    except Exception as exc:
        return jsonify(error=f'Bad image: {exc}'), 400

    # Run SR
    # sr_img = run_sr(img)
    sr_img = upscale_image_from_path_or_url(img)

    # Encode to base64 <img src="data:...">
    buf = io.BytesIO()
    sr_img[0].save(buf, format='JPEG', quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    data_uri = f'data:image/jpeg;base64,{b64}'

    return jsonify(sr=data_uri)
    

if __name__ == "__main__":
    # Use eventlet for production; disable Flask debug for speed.
    socketio.run(app, host="0.0.0.0", port=5000)
