import cv2

def detect_objects(image_path, classifier_path):
    classifier = cv2.CascadeClassifier(classifier_path)
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    objects = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in objects:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_objects('test.jpg', 'haar-cascade/cascade.xml')
