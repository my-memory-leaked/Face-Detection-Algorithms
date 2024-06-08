import os

def train_classifier(positives_txt, negatives_txt, output_dir, num_pos, num_neg, num_stages):
    """
    Trains an LBP cascade classifier using OpenCV.

    :param positives_txt: File containing positive samples.
    :param negatives_txt: File containing negative samples.
    :param output_dir: Directory to save the trained classifier.
    :param num_pos: Number of positive samples.
    :param num_neg: Number of negative samples.
    :param num_stages: Number of stages for training.
    """
    os.system(f'opencv_traincascade -data {output_dir} -vec {positives_txt} -bg {negatives_txt} '
              f'-numPos {num_pos} -numNeg {num_neg} -numStages {num_stages} -featureType LBP -w 24 -h 24')

if __name__ == "__main__":
    train_classifier('../dataset/positives.txt', '../dataset/negatives.txt', 'lbp-cascade', 900, 500, 10)
