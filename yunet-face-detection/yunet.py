import cv2
import time
import os
from matplotlib import pyplot as plt

# Initialize the face detector
detector = cv2.FaceDetectorYN.create("models/face_detection_yunet_2023mar.onnx", "", (320, 320))

# Directory containing images
image_dir = "./../../datasets/my/"

# Loop through each file in the directory
for filename in os.listdir(image_dir):
    file_path = os.path.join(image_dir, filename)

    # Read image
    img = cv2.imread(file_path)

    # Check if the file is a valid image
    if img is None:
        continue

    # Resize the image to a standard size if it's too large
    max_dim = 1024
    if max(img.shape) > max_dim:
        scaling_factor = max_dim / max(img.shape)
        img = cv2.resize(img, (int(img.shape[1] * scaling_factor), int(img.shape[0] * scaling_factor)))

    # Get image dimensions
    img_W = int(img.shape[1])
    img_H = int(img.shape[0])

    # Save time
    t0 = time.time()

    # Getting the detections
    detector.setInputSize((img_W, img_H))
    detections = detector.detect(img)

    # Calculate inference time
    inf_time = round(time.time() - t0, 3)

    # Print results
    print(f"Processing {filename}")
    print(f"Detections: {detections}")
    print(f"Inference time: {inf_time}s")

    # Draw detections
    if (detections[1] is not None) and (len(detections[1]) > 0):
        for detection in detections[1]:
            # Converting predicted and ground truth bounding boxes to required format
            pred_bbox = detection
            pred_bbox = [int(i) for i in pred_bbox[:4]]

            cv2.rectangle(img, pred_bbox, (0, 255, 0), 5)

    # Write inference time
    img = cv2.putText(img, f"Inf Time: {inf_time}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)

    # Save the image with detections
    # output_path = os.path.join(image_dir, f"processed_{filename}")
    cv2.imwrite(output_path, img)

    # Display the image (optional)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
