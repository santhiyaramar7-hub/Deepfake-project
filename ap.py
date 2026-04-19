from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model("deepfake_model.h5")
IMG_SIZE = 64

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    if file.filename == '':
        return "No file selected"

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    print("Saved:", filepath)

    # 🎥 VIDEO → FRAME EXTRACTION
    cap = cv2.VideoCapture(filepath)

    frames = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret or count > 20:   # take only 20 frames
            break

        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = frame / 255.0
        frames.append(frame)
        count += 1

    cap.release()

    if len(frames) == 0:
        return "Error: Cannot read video"

    frames = np.array(frames)

    # 🤖 PREDICTION
    preds = model.predict(frames)

    avg_pred = np.mean(preds, axis=0)

    result = "REAL" if np.argmax(avg_pred) == 0 else "FAKE"
    confidence = round(np.max(avg_pred) * 100, 2)

    return render_template('index.html', 
                       result=result, 
                       confidence=confidence,
                       video_path=filepath)

if __name__ == "__main__":
    app.run(debug=True)
