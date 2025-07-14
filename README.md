[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](https://aihub.qualcomm.com)

<!-- # [Qualcomm® AI Hub Models](https://aihub.qualcomm.com/) -->

[![Release](https://img.shields.io/github/v/release/quic/ai-hub-models)](https://github.com/quic/ai-hub-models/releases/latest)
[![Tag](https://img.shields.io/github/v/tag/quic/ai-hub-models)](https://github.com/quic/ai-hub-models/releases/latest)
[![PyPi](https://img.shields.io/pypi/v/qai-hub-models)](https://pypi.org/project/qai-hub-models/)
![Python 3.9, 3.10, 3.11, 3.12](https://img.shields.io/badge/python-3.9%2C%203.10%20(Recommended)%2C%203.11%2C%203.12-yellow)

<!-- The Qualcomm® AI Hub Models are a collection of
state-of-the-art machine learning models optimized for deployment on Qualcomm® devices.  -->

<!-- * [List of Models by Category](#model-directory)
* [On-Device Performance Data](https://aihub.qualcomm.com/models)
* [Device-Native Sample Apps](https://github.com/quic/ai-hub-apps)

See supported: [On-Device Runtimes](#on-device-runtimes), [Hardware Targets & Precision](#device-hardware--precision), [Chipsets](#chipsets), [Devices](#devices)

&nbsp;

![Demo](https://user-images.githubusercontent.com/demo/face_enhancement.gif)  
A real-time, browser-based face enhancement app using lightweight detection and super-resolution models. Designed for fast inference and educational demonstration, it showcases low-latency AI inference powered by edge devices. -->

# No Brain, Still Sharp — Real-Time Face Enhancement

## Overview

**"No Brain, Still Sharp"** is a real-time face and object detection and enhancement demo. It captures webcam frames in the browser, detects faces and objects, applies super-resolution (SR), and displays cropped enhanced faces and objects live on a responsive frontend. Users can also recover an enhanced full image view by clicking the **"Snap & Enhance"** button.

This app was developed as a Qualcomm AI Edge demo to highlight:

- Lightweight face and object detection (via DETR and custom models)
- Face super-resolution enhancement (ESRGAN-style)
- Seamless real-time experience using Flask and SocketIO

---


## Setup Instructionss

### 1. Clone the repository

```bash
git clone https://github.com/CarlosChen1126/no_brain_still_sharp.git
cd no_brain_still_sharp
```
### 2. Setup

#### Install Python Package

The package is available via pip:

```shell
pip install qai_hub_models
```

---

##  Run & Usage

### 1. Start the backend server

```bash
cd app
python app.py
```

### 2. Open the frontend

Open your browser and go to:  
 `http://127.0.0.1:5000`

### 3. Use the app

- The app automatically starts webcam capture.
- Detected faces and objects will appear live on the right panel.
- Click **"Snap & Enhance"** to generate a full enhanced image using the best resolution.
- Use the filter buttons (e.g. People, Animals, Vehicles) to sort detected objects.
- Click any thumbnail to preview it larger.

 Try to follow the interactions as demonstrated in the [demo slides](https://www.canva.com/design/DAGs_lJj3Jw/2Mro4jSXDx6zlzmDv9vYAg/edit?ui=eyJBIjp7fX0).

---
## Reference
[Qualcomm® AI Hub Models](https://github.com/quic/ai-hub-models)

---

## Developers

| Name           | Contact                        |
|----------------|--------------------------------|
| Morris Fan     | https://github.com/thisismorris|
| ChiRay Tseng   | https://github.com/c1tseng      |
| IChen Chuang   | https://github.com/zhuangggg   |
| Yu-Chen Chen   | https://github.com/carloschen1126|
| Kate Peterson  | katepaigepeterson@gmail.com     |

