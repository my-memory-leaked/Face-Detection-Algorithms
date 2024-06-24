import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score
from skimage.metrics import structural_similarity as ssim
import cv2
from PIL import Image
from collections import defaultdict

# Initialize the YuNet face detector
detector = cv2.FaceDetectorYN.create("models/face_detection_yunet_2023mar.onnx", "", (160, 160))

# Function to calculate IoU
def calculate_iou(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # Compute the area of the intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # Compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

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

# Function to compare detected faces
def compare_faces(yunet_faces, detected_faces, threshold=0.15):
    y_true, y_pred = [], []
    for yunet_face, detected_face in zip(yunet_faces, detected_faces):
        yunet_face_array = np.asarray(yunet_face)
        detected_face_array = np.asarray(detected_face)

        if yunet_face_array.shape[0] < 7 or yunet_face_array.shape[1] < 7 or detected_face_array.shape[0] < 7 or detected_face_array.shape[1] < 7:
            print("Skipping comparison due to small face size.")
            y_true.append(1)
            y_pred.append(0)  # Assign 0 if we cannot calculate SSIM
            continue

        try:
            min_size = min(yunet_face_array.shape[:2])
            win_size = min(7, min_size)
            similarity = ssim(yunet_face_array, detected_face_array, win_size=win_size, channel_axis=2)
            y_true.append(1)
            y_pred.append(1 if similarity > threshold else 0)
        except ValueError as e:
            print(f"Skipping comparison due to error in SSIM calculation: {e}")
            y_true.append(1)
            y_pred.append(0)  # Assign 0 if there is an error

        # print(f'SSIM similarity: {similarity}, y_true: {y_true[-1]}, y_pred: {y_pred[-1]}')

    return y_true, y_pred

# Load YuNet detected faces
def load_yunet_faces(dir):
    faces = []
    class_stats = defaultdict(lambda: {"success": 0, "failure": 0})
    for root, _, files in os.walk(dir):
        for filename in files:
            path = os.path.join(root, filename)
            if not filename.lower().endswith(('png', 'jpg', 'jpeg')):
                continue
            try:
                face = extract_face(path)
                faces.append(face)
                class_name = os.path.basename(root)
                class_stats[class_name]["success"] += 1
            except ValueError as e:
                print(e)
                class_name = os.path.basename(root)
                class_stats[class_name]["failure"] += 1
    return faces, class_stats

# Load detected faces from cropped dataset
def load_cropped_faces(dir):
    faces = []
    for root, _, files in os.walk(dir):
        for filename in files:
            path = os.path.join(root, filename)
            if not filename.lower().endswith(('png', 'jpg', 'jpeg')):
                continue
            image = Image.open(path)
            face = np.asarray(image)
            faces.append(face)
    return faces

# Directories containing original and cropped faces
original_faces_dir = '../../datasets/faces_yunet/Original Images/Original Images'
cropped_faces_dir = '../../datasets/faces_yunet/Faces/Faces'

# Load faces
yunet_faces, class_stats = load_yunet_faces(original_faces_dir)
detected_faces = load_cropped_faces(cropped_faces_dir)

# Print class-wise statistics
print("\nClass-wise face detection statistics:")
for class_name, stats in class_stats.items():
    print(f"Class: {class_name}, Success: {stats['success']}, Failure: {stats['failure']}")

# Compare faces and calculate metrics
y_true, y_pred = compare_faces(yunet_faces, detected_faces)

# Check if there are valid comparisons
if len(y_true) == 0 or len(y_pred) == 0:
    print("No valid face comparisons available.")
else:
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    print(f'\nOverall Metrics:')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
