import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from collections import defaultdict


def read_images_from_folder2(folder):
    label_counts = defaultdict(int)
    images = []
    labels = []

    # First pass: count the number of images per label
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            label = filename.split('_')[0]
            label_counts[label] += 1

    # Second pass: append images and labels that meet the count threshold
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            label = filename.split('_')[0]
            if label_counts[label] >= 10:
                img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Optional: Apply bilateral filter and resize
                    # blurred_img = cv2.bilateralFilter(img, 9, 75, 75)
                    # blurred_img = cv2.resize(blurred_img, (100, 100))  # Resize to a consistent size
                    images.append(img)
                    labels.append(label)

    return images, labels

def read_images_from_folder3(folder, train_labels):
    label_counts = defaultdict(int)
    images = []
    labels = []

    # First pass: count the number of images per label
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            label = filename.split('_')[0]
            label_counts[label] += 1

    # Second pass: append images and labels that meet the count threshold
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            label = filename.split('_')[0]
            if label in train_labels:
                img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Optional: Apply bilateral filter and resize, results are worse
                    # blurred_img = cv2.bilateralFilter(img, 9, 75, 75)
                    # blurred_img = cv2.resize(blurred_img, (100, 100))  # Resize to a consistent size
                    images.append(img)
                    labels.append(label)

    return images, labels


train_folder = '../zdjecia/divided/train'
test_folder = '../zdjecia/divided/test'

train_images, train_labels = read_images_from_folder2(train_folder)
test_images, test_labels = read_images_from_folder3(test_folder, train_labels)

# Resize all images to a common size
image_size = (100, 100)  # You can adjust this size as needed
train_images = [cv2.resize(img, image_size) for img in train_images]
test_images = [cv2.resize(img, image_size) for img in test_images]


# weak results
def extract_lbp_features(image, num_points=24, radius=8):
    lbp = local_binary_pattern(image, num_points, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    return lbp_hist


def extract_hog_features(image):
    # Compute HOG features
    # fd, hog_image = hog(image, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualize=True,) # 72
    # fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True,) # 74
    # fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True,) # 68

    fd = hog(image, orientations=12, pixels_per_cell=(8, 8), cells_per_block=(3, 3), visualize=False, )  # 75
    return fd

# Flatten images to 1D vectors
# train_data = np.array([img.flatten() for img in train_images])
# test_data = np.array([img.flatten() for img in test_images])

# train_data = np.array([extract_lbp_features(img) for img in train_images])
# test_data = np.array([extract_lbp_features(img)  for img in test_images])

train_data = np.array([extract_hog_features(img) for img in train_images])
test_data = np.array([extract_hog_features(img)  for img in test_images])

# Encode labels to integers
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
test_labels = label_encoder.transform(test_labels)

# Compute Fisherfaces
num_classes = len(np.unique(train_labels))
print(num_classes)
num_features = train_data.shape[1]
print(num_features)
num_components = min(num_features, num_classes-1)
# num classes jest 31, wiec 30 njelpszy wynik


# prepare LDA then use classifier SVM or KNN
lda = LinearDiscriminantAnalysis(n_components=num_components)
train_fisherfaces = lda.fit_transform(train_data, train_labels)

# use SVM
classifier = SVC(kernel='linear', C=0.01)
classifier.fit(train_fisherfaces, train_labels)


# use KNN
# classifier = KNeighborsClassifier(n_neighbors=8)  # You can adjust the number of neighbors (k) as needed
# classifier.fit(train_fisherfaces, train_labels)

# Transform test data using the same LDA
test_fisherfaces = lda.transform(test_data)


predictions = classifier.predict(test_fisherfaces)
# Count the number of correct recognitions
# Predict and evaluate

correct_recognitions = sum(
    [1 for recognized, actual in zip(predictions,test_labels) if recognized == actual])

print(f"Correct recognitions: {correct_recognitions}")
print(f"Total images: {len(test_labels)}")
print(f"Recognition accuracy: {correct_recognitions / len(test_labels) * 100:.2f}%")


accuracy_svm = accuracy_score(test_labels, predictions)
precision_svm = precision_score(test_labels, predictions, average='weighted')
recall_svm = recall_score(test_labels, predictions, average='weighted')
f1_svm = f1_score(test_labels, predictions, average='weighted')

print(f"Accuracy: {accuracy_svm:.4f}")
print(f"Precision: {precision_svm:.34f}")
print(f"Recall: {recall_svm:.4f}")
print(f"F1-score: {f1_svm:.4f}")


