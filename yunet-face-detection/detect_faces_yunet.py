import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from keras.models import load_model
from PIL import Image
import os
import logging
import time
from sklearn.metrics import precision_score, recall_score, f1_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the YuNet face detector
detector = cv2.FaceDetectorYN.create("models/face_detection_yunet_2023mar.onnx", "", (320, 320))

# Function to extract a single face from a given photograph using YuNet
def extract_face(filename, required_size=(160, 160)):
    if not os.path.exists(filename):
        logger.error(f"File not found: {filename}")
        raise FileNotFoundError(f"File not found: {filename}")

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

# Load all faces in a directory and save the detected faces in another directory
def process_and_save_faces(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    stats = {
        'class': [],
        'detected_faces': [],
        'undetected_faces': [],
    }

    for root, dirs, files in os.walk(input_dir):
        class_name = os.path.basename(root)
        detected_count = 0
        undetected_count = 0

        for filename in files:
            input_path = os.path.join(root, filename)
            relative_path = os.path.relpath(root, input_dir)
            output_path = os.path.join(output_dir, relative_path)
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            try:
                face = extract_face(input_path)
                output_file = os.path.join(output_path, filename)
                Image.fromarray(face).save(output_file)
                logger.info(f"Processed and saved face for {filename}")
                detected_count += 1
            except ValueError as e:
                logger.warning(e)
                undetected_count += 1

        stats['class'].append(class_name)
        stats['detected_faces'].append(detected_count)
        stats['undetected_faces'].append(undetected_count)

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(os.path.join(output_dir, 'detection_stats.csv'), index=False)

# Calculate performance metrics
def calculate_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    return precision, recall, f1

# Main execution
input_dir = '../../datasets/faces_yunet/Original Images/Original Images'
output_dir = '../../datasets/faces_yunet/detected_faces'

process_and_save_faces(input_dir, output_dir)

# Dummy example of metrics calculation (You need actual ground truth and predictions for this part)
# y_true = [true_labels]  # Ground truth labels
# y_pred = [predicted_labels]  # Predicted labels
# precision, recall, f1 = calculate_metrics(y_true, y_pred)
# logger.info(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

logger.info("Face detection and saving process completed.")
