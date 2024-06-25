import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def load_images_and_labels(base_path):
    images = []
    labels = []
    label_dict = {}
    current_label = 0

    for root, dirs, files in os.walk(base_path):
        for name in files:
            if name.lower().endswith(('.jpeg', '.jpg', '.png')): 
                img_path = os.path.join(root, name)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    images.append(image)
                    label = os.path.basename(root)
                    if label not in label_dict:
                        label_dict[label] = current_label
                        current_label += 1
                    labels.append(label_dict[label])
                else:
                    print(f'Failed to load image: {img_path}')
    
    print(f'Total images loaded: {len(images)}')
    return images, labels

def extract_lbp_features(images, num_points, radius):
    lbp_features = []
    for idx, image in enumerate(images):
        lbp = local_binary_pattern(image, num_points, radius, method='uniform')
        hist, _ = np.histogram(lbp, bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        lbp_features.append(hist)
        if idx % 50 == 0:
            print(f'Processed {idx + 1}/{len(images)} images for LBP features')
    return np.array(lbp_features)

# Ścieżka do katalogu z obrazami testowymi
test_base_path = 'dataset/Original-Images'  # Zmień na odpowiednią ścieżkę

# Parametry LBP używane podczas trenowania modelu
num_points = 24
radius = 2

# Ładowanie obrazów testowych i etykiet
test_images, test_labels = load_images_and_labels(test_base_path)

if len(test_images) == 0 or len(test_labels) == 0:
    raise ValueError('No test images or labels found. Please check the dataset directory and image formats.')

# Ekstrakcja cech LBP z obrazów testowych
test_lbp_features = extract_lbp_features(test_images, num_points, radius)

# Normalizacja cech za pomocą StandardScaler
scaler = StandardScaler()
test_lbp_features = scaler.fit_transform(test_lbp_features)

# Załadowanie wytrenowanego modelu SVM
clf = joblib.load('lbp_svm_model.pkl')

# Predykcja na zbiorze testowym
y_pred = clf.predict(test_lbp_features)

# Wyświetlenie wyników
accuracy = accuracy_score(test_labels, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Ustalanie liczby unikalnych klas
unique_labels = np.unique(test_labels)
target_names = [f'class_{label}' for label in unique_labels]

print("Classification Report:")
print(classification_report(test_labels, y_pred, target_names=target_names))

print("Confusion Matrix:")
print(confusion_matrix(test_labels, y_pred))
