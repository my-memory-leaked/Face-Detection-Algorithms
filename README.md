
# Face-Detection-Algorithms

## Overview

This project is a comprehensive study and implementation of various face detection algorithms. It is a collaborative effort by a team of students aiming to compare and analyze the performance of different face recognition techniques.

## Table of Contents

- [Overview](#overview)
- [Algorithms](#algorithms)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contributors](#contributors)

## Algorithms

The project includes implementations and evaluations of the following face detection algorithms:

- **EigenFaces** (Hania)
- **FisherFaces** (Hania)
- **VGG16** (Marta)
- **Haar Cascade Classifier** (Karol)
- **Local Binary Patterns (LBP)** (Karol)
- **YuNet** (Szymon)
- **FaceNet** (Szymon)

## Project Structure

FaceDetectionAlgorithms/ </br>
├── dataset/ </br>
│ └── (contains datasets for training and testing) </br>
├── eigenface.py </br>
├── fisherface.py </br>
├── vgg16.py </br>
├── haar-cascade-project/ </br>
│ └── (Haar Cascade implementation) </br>
├── local-binary-patterns-project/ </br>
│ └── (LBP implementation) </br>
├── yunet-face-detection/ </br>
│ └── (YuNet implementation) </br>
├── facenet-face-recognition/ </br>
│ └── (FaceNet implementation) </br>
├── tools/ </br>
│ └── (contains utility tools) </br>
├── .gitignore </br>
├── README.md </br>
└── LICENSE </br>

## Installation

To get started with the project, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/yourusername/FaceDetectionAlgorithms.git
```

Make sure you have Python 3.x and pip installed.

## Usage

To run each algorithm, use the corresponding script. For example:

```bash
python eigenface.py
```

Make sure to update the paths to the datasets in each script if necessary.

## Contributors

- **Hania** - EigenFaces, FisherFaces
- **Marta** - VGG16
- **Karol** - Haar Cascade Classifier, Local Binary Patterns
- **Szymon** - YuNet, FaceNet

