import cv2
import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Load Haar cascade models
model1 = cv2.CascadeClassifier('D:/Repositories/wkiro-2024/haar-cascade-project/haar-cascade/differ/opencv-haar.xml')
model2 = cv2.CascadeClassifier('D:/Repositories/wkiro-2024/haar-cascade-project/haar-cascade/differ/own-haar.xml')

# Define a function to calculate accuracy
def calculate_accuracy(detected_objects, ground_truth_objects):
    correct_detections = 0
    for gt in ground_truth_objects:
        for det in detected_objects:
            if iou(det, gt) > 0.5:  # IOU threshold of 0.5 for considering as correct detection
                correct_detections += 1
                break
    accuracy = correct_detections / len(ground_truth_objects) if len(ground_truth_objects) > 0 else 0
    return accuracy

# Function to calculate Intersection over Union (IoU)
def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

# Function to read ground truth data from a file
def read_ground_truth(ground_truth_file):
    ground_truth_data = {}
    with open(ground_truth_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            filename = os.path.basename(parts[0])
            boxes = []
            for i in range(2, len(parts), 4):
                x, y, w, h = map(int, parts[i:i+4])
                boxes.append((x, y, w, h))
            if filename in ground_truth_data:
                ground_truth_data[filename].extend(boxes)
            else:
                ground_truth_data[filename] = boxes
    return ground_truth_data

# Define paths
dataset_path = 'D:/Repositories/wkiro-2024/haar-cascade-project/haar-cascade/differ/pic'
ground_truth_path = 'D:/Repositories/wkiro-2024/haar-cascade-project/haar-cascade/differ/positives.txt'

# Read ground truth data
ground_truth_data = read_ground_truth(ground_truth_path)

# Initialize lists to collect features and labels for SVM
features = []
labels = []
fixed_size = (64, 64)  # Fixed size for ROI

# Iterate through folders and images
for root, dirs, files in os.walk(dataset_path):
    for file_name in files:
        if file_name.endswith('.jpg'):
            file_path = os.path.join(root, file_name)
            image = cv2.imread(file_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            detected_objects_model1 = model1.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            detected_objects_model2 = model2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            # Retrieve ground truth data
            ground_truth = ground_truth_data.get(file_name, [])

            # Calculate accuracy
            accuracy_model1 = calculate_accuracy(detected_objects_model1, ground_truth)
            accuracy_model2 = calculate_accuracy(detected_objects_model2, ground_truth)

            # Print results for the current image
            print(f"Image: {file_name}")
            print(f"Ground Truth: {ground_truth}")
            print(f"Detected by opencv-haar.xml: {detected_objects_model1}")
            print(f"Detected by own-haar.xml: {detected_objects_model2}")
            print(f"Accuracy of opencv-haar.xml model: {accuracy_model1 * 100:.2f}%")
            print(f"Accuracy of own-haar.xml model: {accuracy_model2 * 100:.2f}%")
            print("\n" + "-"*50 + "\n")

            # Extract features for SVM training
            for (x, y, w, h) in detected_objects_model1:
                roi = gray[y:y+h, x:x+w]
                roi_resized = cv2.resize(roi, fixed_size)
                features.append(roi_resized.flatten())
                labels.append(1)  # Assuming 1 for detected objects by model1

            for (x, y, w, h) in detected_objects_model2:
                roi = gray[y:y+h, x:x+w]
                roi_resized = cv2.resize(roi, fixed_size)
                features.append(roi_resized.flatten())
                labels.append(0)  # Assuming 0 for detected objects by model2

# Convert lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Train and evaluate SVM model with GridSearchCV for hyperparameter tuning
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}
grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=2)
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)

# Calculate and print SVM performance metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['model2', 'model1'])
print(f'SVM Accuracy: {accuracy * 100:.2f}%')
print('Classification Report:\n', report)
