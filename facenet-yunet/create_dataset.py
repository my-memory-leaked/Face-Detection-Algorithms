import numpy as np
import pandas as pd
import cv2
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot as plt
from keras.models import load_model
from keras import config as kcfg
from PIL import Image
# import tensorflow as tf
import os
# import logging
# import keras


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tf.get_logger().setLevel('WARNING')
# logging.getLogger('keras').setLevel(logging.WARNING)
# logging.getLogger('tensorflow').setLevel(logging.WARNING)
kcfg.disable_interactive_logging()

# Extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
    # Load image from file
    image = Image.open(filename)
    # Convert to RGB, if needed
    image = image.convert('RGB')
    # Convert to array
    pixels = np.asarray(image)
    # Create the detector, using default weights
    detector = MTCNN()
    # Detect faces in the image
    results = detector.detect_faces(pixels)

    # Check if at least one face was detected
    if len(results) == 0:
        raise ValueError(f"No faces detected in the image: {filename}")

    # Extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # Deal with negative pixel index
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # Extract the face
    face = pixels[y1:y2, x1:x2]
    # Resize pixels to the model size
    image = Image.fromarray(face)
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

# Ensure the directories exist before loading datasets
# train_dir = '../../datasets/faces/train'
# test_dir = '../../datasets/faces/test'

train_dir = '../../datasets/emotions/train'
test_dir = '../../datasets/emotions/test'

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
np.savez_compressed('process/emotions-dataset.npz', trainX, trainy, testX, testy)
