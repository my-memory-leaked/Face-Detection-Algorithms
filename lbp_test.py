import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_images_and_labels(base_path):
    images = []
    labels = []
    label_dict = {}
    current_label = 0

    for root, dirs, files in os.walk(base_path):
        for name in files:
            if name.lower().endswith(('.jpeg', '.jpg')): 
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

base_path = 'dataset/Original-Images'
images, labels = load_images_and_labels(base_path)

if len(images) == 0 or len(labels) == 0:
    raise ValueError('No images or labels found. Please check the dataset directory and image formats.')

# Test specific values for num_points and radius
params = {
    'num_points': [24], 
    'radius': [2]
}

best_accuracy = 0
best_params = {}
best_features = None

for num_points in params['num_points']:
    for radius in params['radius']:
        print(f'Testing LBP with num_points={num_points} and radius={radius}')
        lbp_features = extract_lbp_features(images, num_points, radius)
        
        if lbp_features.size == 0:
            raise ValueError('LBP feature extraction failed. Please check the images.')
        
        X_train, X_test, y_train, y_test = train_test_split(lbp_features, labels, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        clf = SVC(kernel='linear', C=1.0, random_state=42)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy with num_points={num_points} and radius={radius}: {accuracy * 100:.2f}%')
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {'num_points': num_points, 'radius': radius}
            best_features = lbp_features

print(f'Best accuracy: {best_accuracy * 100:.2f}% with params: {best_params}')

# Final training with best parameters
num_points = best_params['num_points']
radius = best_params['radius']
lbp_features = best_features

X_train, X_test, y_train, y_test = train_test_split(lbp_features, labels, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = SVC(kernel='linear', C=1.0, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Final accuracy: {accuracy * 100:.2f}%')

joblib.dump(clf, 'lbp_svm_model.pkl')
print('Model saved to lbp_svm_model.pkl')
