# Haar Classifier Project

## Description
This project demonstrates how to train a Haar cascade classifier for object detection in images using the OpenCV library. It consists of several steps: data preparation, feature vector generation, classifier training, and using the classifier for object detection.

## Project Structure

haar_classifier_project/\
├── dataset/ \
│ ├── positives/\
│ │ ├── positive1.jpg\
│ │ ├── positive2.jpg\
│ │ └── ...\
│ ├── negatives/\
│ │ ├── negative1.jpg\
│ │ ├── negative2.jpg\
│ │ └── ...\
│ ├── positives.txt\
│ └── negatives.txt\
│\
├── haarcascade/\
│ └── cascade.xml\
│\
├── source/\
│ ├── init.py\
│ ├── prepare_data.py\
│ ├── train_classifier.py\
│ └── detect_objects.py\
│\
├── requirements.txt\
├── readme\

### Directory and File Descriptions

- **dataset/**: Directory containing positive and negative images and the respective description files.
  - **positives/**: Directory with positive images.
  - **negatives/**: Directory with negative images.
  - **positives.txt**: File with descriptions of positive images.
  - **negatives.txt**: File with descriptions of negative images.
- **haarcascade/**: Directory where the trained model is stored.
  - **cascade.xml**: File of the trained Haar classifier.
- **source/**: Directory with source code.
  - **prepare_data.py**: Script for data preparation.
  - **train_classifier.py**: Script for training the classifier.
  - **detect_objects.py**: Script for object detection using the trained classifier.
  - **\_\_init\_\_.py**: Initialization file for the module.
- **requirements.txt**: File with the required Python libraries.
- **readme.md**: Project description file.
- **main.py**: Main script for running the various stages of the project.
