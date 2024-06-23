import cv2
import os

def preprocess_image(image, input_size=(320, 320)):
    h, w = image.shape[:2]
    scale = input_size[0] / max(h, w)
    resized_image = cv2.resize(image, (int(w * scale), int(h * scale)))
    pad_image = cv2.copyMakeBorder(resized_image, 0, input_size[0] - resized_image.shape[0], 0, input_size[1] - resized_image.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return pad_image, scale

def prepare_blob(image):
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(320, 320), mean=(104.0, 117.0, 123.0), swapRB=False, crop=False)
    return blob

def detect_faces(net, image):
    input_image, scale = preprocess_image(image)
    blob = prepare_blob(input_image)
    net.setInput(blob)
    detections = net.forward()

    h, w = input_image.shape[:2]
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            box = box.astype(int)
            faces.append(box)
    return faces, scale

def crop_faces(image, faces, scale):
    cropped_faces = []
    for box in faces:
        x, y, x1, y1 = box
        x, y, x1, y1 = int(x / scale), int(y / scale), int(x1 / scale), int(y1 / scale)
        cropped_face = image[y:y1, x:x1]
        cropped_faces.append(cropped_face)
    return cropped_faces

def process_image(image_path, output_dir, net):
    image = cv2.imread(image_path)
    faces, scale = detect_faces(net, image)
    cropped_faces = crop_faces(image, faces, scale)

    for i, face in enumerate(cropped_faces):
        face_filename = os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_face_{i}.jpg")
        cv2.imwrite(face_filename, face)

def process_dataset(dataset_dir, output_dir, net):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            image_path = os.path.join(root, file)
            process_image(image_path, output_dir, net)

def main():
    # Load the pre-trained YuNet model from OpenCV
    model_path = 'moddels/face_detection_yunet_2023mar.onnx'
    net = cv2.dnn.readNet(model_path)

    # Define directories
    train_dir = '../../divided/train'
    test_dir = '../../divided/test'
    output_train_dir = '../../output/train'
    output_test_dir = '../../output/test'

    # Process datasets
    process_dataset(train_dir, output_train_dir, net)
    process_dataset(test_dir, output_test_dir, net)

if __name__ == "__main__":
    main()


