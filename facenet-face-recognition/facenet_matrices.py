from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
from random import choice
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the original dataset to get access to the testX images
original_data = np.load('process/dataset.npz')
trainX, trainy, testX, testy = original_data['arr_0'], original_data['arr_1'], original_data['arr_2'], original_data['arr_3']

# Load the embeddings dataset
data = np.load('process/embeddings.npz')
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

# Predict the classes for the entire test set
yhat_test_all = model.predict(emdTestX_norm)

# Predict probabilities for the entire test set
yhat_prob_all = model.predict_proba(emdTestX_norm)

# Create confusion matrix
cm = confusion_matrix(testy_enc, yhat_test_all)

# Plot and save confusion matrix
plt.figure(figsize=(14, 14))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=out_encoder.classes_, yticklabels=out_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('results/confusion_matrix.png')
plt.close()

# Create probability matrix
prob_matrix = np.zeros((len(out_encoder.classes_), len(out_encoder.classes_)))
for true_label in range(len(out_encoder.classes_)):
    mask = (testy_enc == true_label)
    prob_matrix[true_label, :] = yhat_prob_all[mask].mean(axis=0)

plt.figure(figsize=(20, 20))
sns.heatmap(prob_matrix, annot=True, fmt='.2f', cmap='Reds', xticklabels=out_encoder.classes_, yticklabels=out_encoder.classes_)
plt.xlabel('Predicted Probabilities')
plt.ylabel('True Labels')
plt.title('Probability Matrix')
plt.savefig('results/probability_matrix.png')
plt.close()


# Apply logarithmic scale to the probability matrix
log_prob_matrix = np.log1p(prob_matrix)  # log1p is used to handle zero values

# Plot and save probability matrix with logarithmic scale
plt.figure(figsize=(20, 20))
sns.heatmap(log_prob_matrix, annot=True, fmt='.2f', cmap='Reds', xticklabels=out_encoder.classes_, yticklabels=out_encoder.classes_)
plt.xlabel('Predicted Probabilities (log scale)')
plt.ylabel('True Labels')
plt.title('Probability Matrix (Log Scale)')
plt.savefig('results/probability_matrix_log.png')
plt.close()

