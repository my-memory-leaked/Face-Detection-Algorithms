import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from keras.models import load_model
from keras import config as kcfg
from PIL import Image
import os
import time

# Disable interactive logging
kcfg.disable_interactive_logging()

# Initialize the YuNet face detector
detector = cv2.FaceDetectorYN.create("models/face_detection_yunet_2023mar.onnx", "", (320, 320))

# Function to extract a single face from a given photograph using YuNet
def extract_face(filename, required_size=(160, 160)):
    # Load image from file
    image = Image.open(filename)
    # Convert to RGB, if needed
    image = image.convert('RGB')
    # Convert to array
    pixels = np.asarray(image)
    # Convert to BGR for OpenCV
    img = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

    # Resize the image to a standard size if it's too large
    max_dim = 1024
    if max(img.shape) > max_dim:
        scaling_factor = max_dim / max(img.shape)
        img = cv2.resize(img, (int(img.shape[1] * scaling_factor), int(img.shape[0] * scaling_factor)))

    # Get image dimensions
    img_W = int(img.shape[1])
    img_H = int(img.shape[0])

    # Detect faces using YuNet
    detector.setInputSize((img_W, img_H))
    detections = detector.detect(img)

    # Check if at least one face was detected
    if (detections[1] is None) or (len(detections[1]) == 0):
        raise ValueError(f"No faces detected in the image: {filename}")

    # Extract the bounding box from the first face
    detection = detections[1][0]
    x1, y1, width, height = detection[:4]
    x1, y1, width, height = int(x1), int(y1), int(width), int(height)
    x2, y2 = x1 + width, y1 + height

    # Ensure the coordinates are within the image dimensions
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_W, x2), min(img_H, y2)

    # Extract the face
    face = img[y1:y2, x1:x2]
    # Resize pixels to the model size
    image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array

# Load all faces in a directory
def load_face(dir):
    faces = list()
    # Enumerate files
    for filename in os.listdir(dir):
        path = os.path.join(dir, filename)
        try:
            face = extract_face(path)
            faces.append(face)
        except ValueError as e:
            print(e)
    return faces

# Load dataset
def load_dataset(dir):
    # Lists for faces and labels
    X, y = list(), list()
    for subdir in os.listdir(dir):
        path = os.path.join(dir, subdir)
        if os.path.isdir(path):
            faces = load_face(path)
            labels = [subdir for i in range(len(faces))]
            print(f"Loaded {len(faces)} samples for class: {subdir}")  # Print progress
            X.extend(faces)
            y.extend(labels)
    return np.asarray(X), np.asarray(y)

train_dir = '../../datasets/faces/train'
test_dir = '../../datasets/faces/test'

if os.path.exists(train_dir):
    trainX, trainy = load_dataset(train_dir)
    print(trainX.shape, trainy.shape)
else:
    print(f"Training directory does not exist: {train_dir}")

if os.path.exists(test_dir):
    testX, testy = load_dataset(test_dir)
    print(testX.shape, testy.shape)
else:
    print(f"Testing directory does not exist: {test_dir}")

# Save and compress the dataset for further use
np.savez_compressed('process/faces-dataset.npz', trainX, trainy, testX, testy)
