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

    total_images = 0
    detected_faces = 0
    undetected_faces = 0

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
                detected_faces += 1
            except ValueError as e:
                logger.warning(e)
                undetected_count += 1
                undetected_faces += 1

            total_images += 1

        stats['class'].append(class_name)
        stats['detected_faces'].append(detected_count)
        stats['undetected_faces'].append(undetected_count)

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(os.path.join(output_dir, 'detection_stats.csv'), index=False)

    # Calculate and log accuracy, precision, recall, and F1-score
    accuracy = detected_faces / total_images
    precision = detected_faces / (detected_faces + 0)  # No false positives in this context
    recall = detected_faces / (detected_faces + undetected_faces)
    f1 = 2 * (precision * recall) / (precision + recall)

    logger.info(f"Detection Accuracy: {accuracy:.5f}")
    logger.info(f"Detection Precision: {precision:.5f}")
    logger.info(f"Detection Recall: {recall:.5f}")
    logger.info(f"Detection F1 Score: {f1:.5f}")

# Main execution
input_dir = '../../datasets/faces_yunet/Original Images/Original Images'
output_dir = '../../datasets/faces_yunet/out'

process_and_save_faces(input_dir, output_dir)

logger.info("Face detection and saving process completed.")
