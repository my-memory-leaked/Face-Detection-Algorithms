import numpy as np
import tensorflow as tf
from keras.models import load_model

# load the face dataset
data = np.load('../yunet-face-detection/process/emotions-dataset-yunet.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)

try:
    # load the facenet model
    # facenet_model = load_model(model_path)
    import models.inception_resnet_v1 as inception_resnet_v1
    facenet_model = inception_resnet_v1.InceptionResNetV1()
    facenet_model.load_weights('models/facenet_keras_weights.h5')
    print('Loaded Model')
except Exception as e:
    print(f"Error loading model: {e}")
    raise

def get_embedding(model, face):
    # scale pixel values
    face = face.astype('float32')
    # standardization
    mean, std = face.mean(), face.std()
    face = (face-mean)/std
    # transfer face into one sample (3 dimension to 4 dimension)
    sample = np.expand_dims(face, axis=0)
    # make prediction to get embedding
    yhat = model.predict(sample)
    return yhat[0]

# convert each face in the train set into embedding
emdTrainX = list()
for face in trainX:
    emd = get_embedding(facenet_model, face)
    emdTrainX.append(emd)

emdTrainX = np.asarray(emdTrainX)
print(emdTrainX.shape)

# convert each face in the test set into embedding
emdTestX = list()
for face in testX:
    emd = get_embedding(facenet_model, face)
    emdTestX.append(emd)
emdTestX = np.asarray(emdTestX)
print(emdTestX.shape)

# save arrays to one file in compressed format
np.savez_compressed('process/emotions-embeddings-yunet.npz', emdTrainX, trainy, emdTestX, testy)
