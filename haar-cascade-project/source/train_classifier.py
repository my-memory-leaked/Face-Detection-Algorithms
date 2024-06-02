import os

def train_classifier(positives_vec, negatives_txt, output_dir, num_pos, num_neg, num_stages):
    """
    Trains a Haar cascade classifier using OpenCV.

    :param positives_vec: File containing positive samples in vector format.
    :param negatives_txt: File containing negative samples.
    :param output_dir: Directory to save the trained classifier.
    :param num_pos: Number of positive samples.
    :param num_neg: Number of negative samples.
    :param num_stages: Number of stages for training.
    """
    os.system(f'opencv_traincascade -data {output_dir} -vec {positives_vec} -bg {negatives_txt} '
              f'-numPos {num_pos} -numNeg {num_neg} -numStages {num_stages} -w 24 -h 24')

if __name__ == "__main__":
    train_classifier('dataset/positives.vec', 'dataset/negatives.txt', 'haar-cascade', 900, 500, 10)
