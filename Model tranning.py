import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, GlobalAveragePooling2D

# Define paths and parameters
video_dir = r"C:\Users\janar\Downloads\Violence detection system\dataset"  # Directory containing videos
labels = {"fight": 0, "no_fight": 1}  # Replace with your class names
frame_size = (224, 224)  # Frame size for MobileNetV2
sequence_length = 30  # Number of frames per sequence
batch_size = 8

def extract_frames(video_path, sequence_length, frame_size):
    """Extracts evenly spaced frames from a video."""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, frame_count - 1, sequence_length, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_size)
        frame = frame / 255.0  # Normalize
        frames.append(frame)

    cap.release()
    return np.array(frames) if len(frames) == sequence_length else None

def load_data(video_dir, labels, sequence_length, frame_size):
    """Loads video data and corresponding labels."""
    X, y = [], []
    for label, class_id in labels.items():
        class_dir = os.path.join(video_dir, label)
        for video_file in os.listdir(class_dir):
            video_path = os.path.join(class_dir, video_file)
            frames = extract_frames(video_path, sequence_length, frame_size)
            if frames is not None:
                X.append(frames)
                y.append(class_id)

    return np.array(X), np.array(y)

# Load data
X, y = load_data(video_dir, labels, sequence_length, frame_size)

# Shuffle and split data
indices = np.arange(len(X))
np.random.shuffle(indices)
X, y = X[indices], y[indices]
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the model
feature_extractor = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extractor.trainable = False

model = Sequential([
    TimeDistributed(feature_extractor, input_shape=(sequence_length, *frame_size, 3)),
    TimeDistributed(GlobalAveragePooling2D()),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=10)

# Save the model
model.save("video_classification_model.h5")
