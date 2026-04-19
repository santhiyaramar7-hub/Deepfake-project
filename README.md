# Deepfake Detection Web App

A professional-grade web application built for detecting deepfake videos using a Convolutional Neural Network (CNN). Designed with a polished UI and integrated backend powered by FastAPI, this project was developed for hackathon use cases and practical AI deployment.

---

## 🚀 Features
- **Deepfake Detection**: CNN model trained on Kaggle datasets for video/image classification.
- **FastAPI Backend**: Efficient API endpoints for model inference and file handling.
- **Modern UI/UX**:
  - Drag-and-drop video upload
  - Animated buttons and progress bars
  - Responsive, visually attractive interface
- **Real-time Feedback**: Progress indicators during upload and detection.

---

## 🛠️ Tech Stack
- **Frontend**: HTML, CSS, JavaScript
- **Backend**: FastAPI (Python)
- **Model**: CNN (TensorFlow/Keras or PyTorch)
- **Deployment**: VS Code / Localhost (extendable to cloud)

---

## 📂 Project Structure
deepfake-detection-app/
│
├── backend/
│   ├── main.py          # FastAPI server
│   ├── model.py         # CNN model loading & inference
│   └── utils.py         # Helper functions
│
├── frontend/
│   ├── index.html       # Main UI
│   ├── style.css        # Styling
│   └── script.js        # Upload & progress logic
│
├── models/
│   └── deepfake_cnn.h5  # Trained model file
│
├── requirements.txt     # Python dependencies
└── README.md
