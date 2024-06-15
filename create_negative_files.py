import cv2
import os

def replace_spaces_with_hyphens(directory):
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in dirs:
            new_name = name.replace(" ", "-")
            if new_name != name:
                old_path = os.path.join(root, name)
                new_path = os.path.join(root, new_name)
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed directory: {old_path} -> {new_path}")
                except Exception as e:
                    print(f"Error renaming directory {old_path} to {new_path}: {e}")
        
        for name in files:
            new_name = name.replace(" ", "-")
            if new_name != name:
                old_path = os.path.join(root, name)
                new_path = os.path.join(root, new_name)
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed file: {old_path} -> {new_path}")
                except Exception as e:
                    print(f"Error renaming file {old_path} to {new_path}: {e}")

def detect_and_save_negatives(base_dir, output_file, classifier_path):
    """
    Detect faces in images and save images without faces as negatives.

    :param base_dir: Base directory containing images in subdirectories.
    :param output_file: Output file to save the list of images without faces.
    :param classifier_path: Path to the pre-trained Haar Cascade XML file.
    """
    classifier = cv2.CascadeClassifier(classifier_path)
    with open(output_file, 'w') as f:
        for subdir, _, files in os.walk(base_dir):
            for filename in files:
                if filename.endswith('.jpg'):
                    image_path = os.path.join(subdir, filename)
                    image = cv2.imread(image_path)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    if len(faces) == 0:  # No faces detected
                        f.write(f'{image_path}\n')

if __name__ == "__main__":
    dataset_dir = 'dataset/Original Images'
    replace_spaces_with_hyphens(dataset_dir)
    detect_and_save_negatives('dataset/Original-Images', 'dataset/negatives.txt', 'haar-cascade-project/haar-cascade/opencv/haarcascade_frontalface_default.xml')
