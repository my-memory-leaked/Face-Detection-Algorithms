import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import joblib
from screeninfo import get_monitors

# Load the trained SVM model
clf = joblib.load('lbp_svm_model.pkl')

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_image(image, num_points=24, radius=8):
    lbp = local_binary_pattern(image, num_points, radius, method='uniform')
    hist, _ = np.histogram(lbp, bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def detect_and_classify_faces(image_path, num_points=24, radius=8):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        features = preprocess_image(face, num_points, radius)
        features = features.reshape(1, -1)
        prediction = clf.predict(features)
        label = prediction[0]
        
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, str(label), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Get screen size
    monitor = get_monitors()[0]
    screen_width = monitor.width
    screen_height = monitor.height

    # Resize the image to fit the screen while maintaining aspect ratio
    height, width = image.shape[:2]
    scaling_factor = min(screen_width / width, screen_height / height)
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    resized_image = cv2.resize(image, (new_width, new_height))

    cv2.imshow('Detected Faces', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example image path
new_image_path = '2.jpg'
detect_and_classify_faces(new_image_path)
