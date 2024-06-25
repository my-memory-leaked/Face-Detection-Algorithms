############# VGG16 #############
import cv2
from keras.applications import VGG16
from keras.layers import *
from keras.models import *
from keras.optimizers import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from statistics import mean
import sys

# Import images from dataset folder
path = 'Faces/'
files = os.listdir(path)
size = len(files)

print("Total samples: ", size)

# Convert images and get assigned face classification
images = []
names = []
classes = []

for file in files:
    image = cv2.imread(path+file,0)
    image = cv2.resize(image,dsize=(160,160))
    image = image.reshape((image.shape[0],image.shape[1],1))
    images.append(image)
    split_var = file.split('_')
    name = split_var[0]
    names.append(name)
    if name not in classes:
        classes.append(name)

# Shuffle the dataset
temp = list(zip(images, names))
random.shuffle(temp)
images, names = zip(*temp)

# Display category sizes
cat_sizes = [0] * len(classes)

for n in names:
    cat_sizes[classes.index(n)] += 1
for c, cl in zip(cat_sizes, classes):
    print("Category '", cl , "': ", c)

# Format dataset
size = len(images)

x = np.zeros((size,160,160,3), dtype='float32')
y = np.zeros((size,1), dtype='float32')

for i in range(size):
    x[i] = images[i]
    y[i] = classes.index(names[i])

x = x / 255

# Split data into train/validation/test sets
x_model, x_test, y_model, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
x_train, x_val, y_train, y_val = train_test_split(x_model, y_model, test_size=0.25, random_state=42, stratify=y_model)

# Prepare model structure
base = VGG16(include_top=False, weights="imagenet", input_tensor=Input(shape=(160,160,3)), pooling=max)

flat = Flatten()(base.output)
vgg16_model = Dense(4096, activation='relu')(flat)
vgg16_model = Dense(4096, activation='relu')(vgg16_model)
vgg16_model = Dense(len(classes), activation='softmax')(vgg16_model)

model = Model(inputs=base.input, outputs=vgg16_model)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train model
h = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20, batch_size=64, shuffle=True)

# Save prepared model to file (not recommended, generates large file)
# model.save('data.h5')

history = h

# Display accuracy graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Classification accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'testing'], loc='lower right')
plt.show()

plt.savefig('acc.png')
plt.clf()

# Display loss graph
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Classification loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'testing'], loc='upper right')
plt.show()

plt.savefig('loss.png')

# Display confusion matrix
y_pred_t = model.predict(x_test)
y_pred = np.argmax(y_pred_t, axis=-1)

f, ax = plt.subplots(1,1,figsize=(20,20))
metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, display_labels = classes)
plt.xticks(rotation=90)
plt.show()

plt.savefig('conf_matrix.png')
plt.clf()

# Display accuracy matrix
f, ax = plt.subplots(1,1,figsize=(20,20))
metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, display_labels = classes, normalize = 'true')
plt.xticks(rotation=90)
plt.show()

plt.savefig('acc_matrix.png')

# Display scores
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred, normalize=True))
print("Precision: ", mean(metrics.precision_score(y_test, y_pred, average=None)))
print("Recall: ", mean(metrics.recall_score(y_test, y_pred, average=None)))
print("F1-score: ", mean(metrics.f1_score(y_test, y_pred, average=None)))
