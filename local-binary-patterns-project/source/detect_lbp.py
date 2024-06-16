import cv2

def detect_objects(image_path, classifier_path, scale_factor=0.5):
    """
    Detects objects in an image using a pre-trained LBP cascade classifier.

    :param image_path: Path to the image file.
    :param classifier_path: Path to the LBP cascade XML file.
    :param scale_factor: Factor by which the image will be scaled down. Default is 0.5.
    """
    classifier = cv2.CascadeClassifier(classifier_path)
    image = cv2.imread(image_path)
    
    if image is None:
        raise IOError(f"Cannot load image from {image_path}")
    
    # Resize the image
    image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    objects = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in objects:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_objects('../../2.jpg', '../local-binary-patterns-cascade/opencv/lbpcascade_frontalface.xml')
