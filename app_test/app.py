from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)
camera = cv2.VideoCapture(0)  # 使用預設攝影機

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # 編碼成 JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # 傳給前端（MIME multipart）
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
