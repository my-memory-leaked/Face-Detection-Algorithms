import numpy as np
import cv2
from matplotlib import pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
from PIL import Image

detector = cv2.FaceDetectorYN.create("../yunet-face-detection/models/face_detection_yunet_2023mar.onnx", "", (320, 320))

import models.inception_resnet_v1 as inception_resnet_v1
facenet_model = inception_resnet_v1.InceptionResNetV1()
facenet_model.load_weights('models/facenet_keras_weights.h5')

original_data = np.load('../yunet-face-detection/process/faces-ours-dataset-yunet.npz')
trainX, trainy, testX, testy = original_data['arr_0'], original_data['arr_1'], original_data['arr_2'], original_data['arr_3']

data = np.load('process/faces-ours-embeddings-yunet.npz')
emdTrainX, trainy, emdTestX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

in_encoder = Normalizer()
emdTrainX_norm = in_encoder.fit_transform(emdTrainX)
emdTestX_norm = in_encoder.transform(emdTestX)

out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy_enc = out_encoder.transform(trainy)
testy_enc = out_encoder.transform(testy)

model = SVC(kernel='poly', class_weight='balanced', degree=5, probability=True)
model.fit(emdTrainX_norm, trainy_enc)

def extract_face(image_path, required_size=(160, 160)):
    image = Image.open(image_path)
    image = image.convert('RGB')
    pixels = np.asarray(image)
    img = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

    max_dim = 1024
    if max(img.shape) > max_dim:
        scaling_factor = max_dim / max(img.shape)
        img = cv2.resize(img, (int(img.shape[1] * scaling_factor), int(img.shape[0] * scaling_factor)))

    img_W = int(img.shape[1])
    img_H = int(img.shape[0])

    detector.setInputSize((img_W, img_H))
    detections = detector.detect(img)

    if (detections[1] is None) or (len(detections[1]) == 0):
        raise ValueError(f"No faces detected in the image: {image_path}")

    detection = detections[1][0]
    x1, y1, width, height = detection[:4]
    x1, y1, width, height = int(x1), int(y1), int(width), int(height)
    x2, y2 = x1 + width, y1 + height

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_W, x2), min(img_H, y2)

    face = img[y1:y2, x1:x2]

    image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array

def get_embedding(model, face):
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    sample = np.expand_dims(face, axis=0)
    yhat = model.predict(sample)
    return yhat[0]

def recognize_person(image_path):
    try:
        face = extract_face(image_path)
        embedding = get_embedding(facenet_model, face)
        embedding_norm = in_encoder.transform([embedding])

        yhat_class = model.predict(embedding_norm)
        yhat_prob = model.predict_proba(embedding_norm)

        class_index = yhat_class[0]
        class_probability = yhat_prob[0, class_index] * 100
        predict_names = out_encoder.inverse_transform(yhat_class)
        all_names = out_encoder.inverse_transform(range(len(out_encoder.classes_)))

        print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
        print('Predicted probabilities:\n%s' % dict(zip(all_names, yhat_prob[0] * 100)))

        plt.imshow(face)
        title = '%s (%.3f)' % (predict_names[0], class_probability)
        plt.title(title)
        plt.show()
    except ValueError as e:
        print(e)

image1 = '/home/szymon/Desktop/Screenshot from 2024-06-25 23-46-01.png'
image2 = '/home/szymon/Desktop/Marta Wojcik_6.jpg'
image3 = '/home/szymon/Desktop/Karol Pisarski_4.jpg'

recognize_person(image1)
recognize_person(image2)
recognize_person(image3)
