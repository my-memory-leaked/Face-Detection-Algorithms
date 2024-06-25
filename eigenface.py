from collections import defaultdict

import numpy as np
import cv2
import os
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def extract_hog_features(image):
    fd = hog(image, orientations=12, pixels_per_cell=(8, 8), cells_per_block=(3, 3), visualize=False, )
    return fd


def process_images(folder):
    images = []
    labels = []
    label_dict = {}
    current_label = 0
    min_images_per_label = 10  # Minimum number of images per label

    # Count the number of images per label
    label_count = {}

    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (100, 100))  # Resize to a consistent size

            # images.append(img.flatten())  # Flatten the image
            images.append(extract_hog_features(img))  # use hog to extract features

            surname = filename.split('_')[0]  # filename format is "surname_xxx.jpg"
            if surname not in label_dict:
                label_dict[surname] = current_label
                current_label += 1

            labels.append(label_dict[surname])

            # Count images per label
            if surname in label_count:
                label_count[surname] += 1
            else:
                label_count[surname] = 1

    # Filter out labels with fewer than min_images_per_label images
    filtered_images = []
    filtered_labels = []
    filtered_label_dict = {}
    new_label_index = 0


    for i, label in enumerate(labels):
        surname = list(label_dict.keys())[list(label_dict.values()).index(label)]
        if label_count[surname] >= min_images_per_label:
            filtered_images.append(images[i])
            if surname not in filtered_label_dict:
                filtered_label_dict[surname] = new_label_index
                new_label_index += 1
            filtered_labels.append(filtered_label_dict[surname])

    print(len(filtered_label_dict))
    return np.array(filtered_images), np.array(filtered_labels), filtered_label_dict


# Step 1: Load the dataset
folder = '../zdjecia/divided/train'
images, labels, label_dict = process_images(folder)

# Step 2: Compute the mean face
mean_face = np.mean(images, axis=0)

# Step 3: Subtract the mean face from all images
mean_subtracted_images = images - mean_face

# Step 4: Perform PCA to get the eigenfaces

# pca = PCA(n_components=200)  # Adjust the number of components as needed  //ok for svm
pca = PCA(n_components=100)  # Adjust the number of components as   // ok for knn


pca.fit(mean_subtracted_images)
eigenfaces = pca.components_


# Step 5: Project the images into the eigenface space
projected_images = pca.transform(mean_subtracted_images)

# Step 6: Train an SVM or KNN classifier on the projected images
scaler = StandardScaler()
projected_images_scaled = scaler.fit_transform(projected_images, labels)

# use svm
classifier = SVC(kernel='linear', C=0.01)
classifier.fit(projected_images_scaled, labels)

# use KNN
# classifier = KNeighborsClassifier(n_neighbors=8)  # You can adjust the number of neighbors (k) as needed
# classifier.fit(projected_images_scaled, labels)


# Step 7: Recognize faces in a given folder
def recognize_faces_in_folder(folder, pca, mean_face, scaler, classifier, labels):
    recognition_results = []
    ground_truth_labels = []

    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)

        # Extract the ground truth label
        actual_surname = filename.split('_')[0]  # filename format is "surname_x.jpg"

        if actual_surname in labels:
            ground_truth_labels.append(actual_surname)
            recognized_person = recognize_face(img_path, pca, mean_face, scaler, classifier, labels)
            recognition_results.append(recognized_person)

    return recognition_results, ground_truth_labels


# Helper function to recognize a single face
def recognize_face(new_face_path, pca, mean_face, scaler, svm, labels):
    new_face = cv2.imread(new_face_path, cv2.IMREAD_GRAYSCALE)

    # new_face = cv2.resize(new_face, (100, 100)).flatten()

    #use hog to extract features
    new_face = cv2.resize(new_face, (100, 100))
    new_face = extract_hog_features(new_face)

    mean_subtracted_new_face = new_face - mean_face
    projected_new_face = pca.transform([mean_subtracted_new_face])
    projected_new_face_scaled = scaler.transform(projected_new_face)

    recognized_label = svm.predict(projected_new_face_scaled)[0]
    recognized_person = list(labels.keys())[list(labels.values()).index(recognized_label)]

    return recognized_person


# Example usage: recognize all faces in a folder and count correct recognitions
recognition_folder = '../zdjecia/divided/test'
recognized_faces, ground_truth_faces = recognize_faces_in_folder(recognition_folder, pca, mean_face, scaler, classifier,
                                                                 label_dict)

# Count the number of correct recognitions
correct_recognitions = sum(
    [1 for recognized, actual in zip(recognized_faces, ground_truth_faces) if recognized == actual])

print(f"Correct recognitions: {correct_recognitions}")
print(f"Total images: {len(ground_truth_faces)}")
print(f"Recognition accuracy: {correct_recognitions / len(ground_truth_faces) * 100:.2f}%")

accuracy_svm = accuracy_score(ground_truth_faces, recognized_faces)
precision_svm = precision_score(ground_truth_faces, recognized_faces, average='weighted')
recall_svm = recall_score(ground_truth_faces, recognized_faces, average='weighted')
f1_svm = f1_score(ground_truth_faces, recognized_faces, average='weighted')

print(f"Accuracy: {accuracy_svm:.5f}")
print(f"Precision: {precision_svm:.5f}")
print(f"Recall: {recall_svm:.5f}")
print(f"F1-score: {f1_svm:.5f}")
