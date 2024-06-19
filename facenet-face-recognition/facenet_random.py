from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from random import choice
import numpy as np
import matplotlib.pyplot as plt

# Load the original dataset to get access to the testX images
original_data = np.load('process/emotions-dataset.npz')
trainX, trainy, testX, testy = original_data['arr_0'], original_data['arr_1'], original_data['arr_2'], original_data['arr_3']

# Load the embeddings dataset
data = np.load('process/emotions-embeddings.npz')
emdTrainX, trainy, emdTestX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

# Normalize input vectors
in_encoder = Normalizer()
emdTrainX_norm = in_encoder.fit_transform(emdTrainX)
emdTestX_norm = in_encoder.transform(emdTestX)

# Label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy_enc = out_encoder.transform(trainy)
testy_enc = out_encoder.transform(testy)

# Fit model
model = SVC(kernel='linear', probability=True)
model.fit(emdTrainX_norm, trainy_enc)

# Select a random face from the test set
selection = choice([i for i in range(emdTestX.shape[0])])
random_face = testX[selection]  # This should be the original testX face image corresponding to the embedding
random_face_emd = emdTestX_norm[selection]
random_face_class = testy_enc[selection]
random_face_name = out_encoder.inverse_transform([random_face_class])

# Prediction for the face
samples = np.expand_dims(random_face_emd, axis=0)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)

# Get name
class_index = yhat_class[0]
class_probability = yhat_prob[0, class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)
all_names = out_encoder.inverse_transform(range(len(out_encoder.classes_)))

print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
print('Predicted probabilities:\n%s' % dict(zip(all_names, yhat_prob[0] * 100)))
print('Expected: %s' % random_face_name[0])

# Plot face
plt.imshow(random_face)
title = '%s (%.3f)' % (predict_names[0], class_probability)
plt.title(title)
plt.show()

# Save the plot as an image file
plt.savefig('results/emotions-plot.png')
plt.close()
