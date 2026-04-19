import tensorflow as tf
from tensorflow.keras import layers, models

# Parameters
IMG_SIZE = 128
SEQUENCE_LENGTH = 10

# CNN model (feature extractor)
cnn = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten()
])

# Full model (CNN + LSTM)
model = models.Sequential([
    layers.TimeDistributed(cnn, input_shape=(SEQUENCE_LENGTH, 128,128,3)),
    layers.LSTM(64, return_sequences=False),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()
import os
import cv2
import numpy as np

def load_data(path):
    X = []
    y = []

    for label in ['real', 'fake']:
        class_path = os.path.join(path, label)
        label_num = 0 if label == 'real' else 1

        for video in os.listdir(class_path):
            video_path = os.path.join(class_path, video)
            frames = []

            for img in sorted(os.listdir(video_path))[:10]:
                img_path = os.path.join(video_path, img)
                frame = cv2.imread(img_path)
                frame = cv2.resize(frame, (128,128))
                frame = frame / 255.0
                frames.append(frame)

            if len(frames) == 10:
                X.append(frames)
                y.append(label_num)

    return np.array(X), np.array(y)

X, y = load_data("C:\Users\santh\OneDrive\Desktop\Deepfake project\model\dataset")

model.fit(X, y, epochs=10, batch_size=4)
model.save("model.h5")
import os
print(os.listdir('C:\Users\santh\OneDrive\Desktop\Deepfake project\model\dataset')
