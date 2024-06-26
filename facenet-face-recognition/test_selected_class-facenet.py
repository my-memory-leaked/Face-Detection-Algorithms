from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

krnl = 'linear'
# krnl = 'sigmoid'
# krnl = 'poly'

# Load the original dataset to get access to the testX images
original_data = np.load('../yunet-face-detection/process/faces-ours-dataset-yunet.npz')
trainX, trainy, testX, testy = original_data['arr_0'], original_data['arr_1'], original_data['arr_2'], original_data['arr_3']

# Load the embeddings dataset
data = np.load('process/faces-ours-embeddings-yunet.npz')
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
# model = SVC(kernel=krnl, probability=True) # linear, standard
# model = SVC(kernel=krnl, class_weight='balanced', probability=True) # sigmoid
# model = SVC(kernel=krnl, probability=True)
model = SVC(kernel='poly', class_weight='balanced', degree=5, probability=True)
model.fit(emdTrainX_norm, trainy_enc)

# Define the name of the person to be searched for
person_name = "Szymon Maciag"  # Replace this with the desired name

# Get the index of the person in the test set
try:
    person_index = list(testy).index(person_name)
except ValueError:
    print(f"Person '{person_name}' not found in the test set.")
    exit()

person_face = testX[person_index]  # This should be the original testX face image corresponding to the embedding
person_face_emd = emdTestX_norm[person_index]
person_face_class = testy_enc[person_index]
person_face_name = out_encoder.inverse_transform([person_face_class])

# Prediction for the face
samples = np.expand_dims(person_face_emd, axis=0)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)

# Get name
class_index = yhat_class[0]
class_probability = yhat_prob[0, class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)
all_names = out_encoder.inverse_transform(range(len(out_encoder.classes_)))

print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
print('Predicted probabilities:\n%s' % dict(zip(all_names, yhat_prob[0] * 100)))
print('Expected: %s' % person_face_name[0])

# Plot face
plt.imshow(person_face)
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
plt.savefig('results/faces-ours-confusion_matrix-'+krnl+'-yunet.png')
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
plt.savefig('results/faces-ours-probability_matrix-'+krnl+'-yunet.png')
plt.close()

# Apply logarithmic scale to the probability matrix
log_prob_matrix = np.log1p(prob_matrix)  # log1p is used to handle zero values

# Plot and save probability matrix with logarithmic scale
plt.figure(figsize=(20, 20))
sns.heatmap(log_prob_matrix, annot=True, fmt='.2f', cmap='Reds', xticklabels=out_encoder.classes_, yticklabels=out_encoder.classes_)
plt.xlabel('Predicted Probabilities (log scale)')
plt.ylabel('True Labels')
plt.title('Probability Matrix (Log Scale)')
plt.savefig('results/faces-ours-probability_matrix_log-'+krnl+'-yunet.png')
plt.close()

# Calculate metrics
accuracy = accuracy_score(testy_enc, yhat_test_all)
precision = precision_score(testy_enc, yhat_test_all, average=None)
recall = recall_score(testy_enc, yhat_test_all, average=None)
f1 = f1_score(testy_enc, yhat_test_all, average=None)

print('Accuracy: %.3f' % accuracy)
for idx, class_name in enumerate(out_encoder.classes_):
    print(f'Class: {class_name}')
    print(f'Precision: {precision[idx]:.5f}')
    print(f'Recall: {recall[idx]:.5f}')
    print(f'F1-score: {f1[idx]:.5f}')
    print('')

precision_weighted = precision_score(testy_enc, yhat_test_all, average='weighted')
recall_weighted = recall_score(testy_enc, yhat_test_all, average='weighted')
f1_weighted = f1_score(testy_enc, yhat_test_all, average='weighted')

print('Weighted Precision: %.5f' % precision_weighted)
print('Weighted Recall: %.5f' % recall_weighted)
print('Weighted F1-score: %.5f' % f1_weighted)

# Full classification report
report = classification_report(testy_enc, yhat_test_all, target_names=out_encoder.classes_)
print(report)
