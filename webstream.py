from flask import Flask, render_template_string, Response
from ultralytics import YOLO
import cv2
import jetson.inference
import jetson.utils
import numpy as np

# Flask app + YOLO model
app = Flask(__name__)
model = YOLO('/home/nvidia02/emotional-detection/runs/classify/train5/weights/best.pt')

# Open webcam
cap = cv2.VideoCapture(0)

# Updated HTML title to "Emotion Detection"
HTML = '''
<!doctype html>
<html>
<head><title>Emotion Detection</title></head>
<body style="text-align:center; background:#f4f4f4; font-family:sans-serif;">
    <h1>Emotion Detection</h1>
    <img src="/video" width="720">
</body>
</html>
'''

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Emotion classification
        result = model.predict(frame, verbose=False)[0]
        top1_id = result.probs.top1
        confidence = float(result.probs.top1conf)
        label = result.names[top1_id].lower()
        percent = int(confidence * 100)

        # Emotion detection logic
        if confidence < 0.65:
            text = "N/A – No Emotion Detected"
            color = (128, 128, 128)  # Gray
        elif "anger" in label:
            text = f"ANGER ({percent}%)"
            color = (255, 0, 0)  # Red
        elif "contempt" in label:
            text = f"CONTEMPT ({percent}%)"
            color = (0, 0, 255)  # Blue
        elif "disgust" in label:
            text = f"DISGUST ({percent}%)"
            color = (0, 255, 0)  # Green
        elif "fear" in label:
            text = f"FEAR ({percent}%)"
            color = (255, 165, 0)  # Orange
        elif "happy" in label:
            text = f"HAPPY ({percent}%)"
            color = (255, 255, 0)  # Yellow
        elif "neutral" in label:
            text = f"NEUTRAL ({percent}%)"
            color = (169, 169, 169)  # Gray
        elif "sad" in label:
            text = f"SAD ({percent}%)"
            color = (0, 0, 0)  # Black
        elif "surprise" in label:
            text = f"SURPRISE ({percent}%)"
            color = (255, 105, 180)  # Pink
        else:
            text = "N/A – Unknown Emotion"
            color = (100, 100, 100)  # Dark Gray

        # Convert to CUDA image 
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cuda_img = jetson.utils.cudaFromNumpy(img_rgb)

        # Convert back to numpy for web stream
        np_img = jetson.utils.cudaToNumpy(cuda_img)
        np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

        # Overlay emotion text
        cv2.putText(np_img, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)

        # Stream over HTTP
        ret, buffer = cv2.imencode('.jpg', np_img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Launch the server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
