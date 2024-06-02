import os

def train_classifier(positives_vec, negatives_txt, output_dir, num_pos, num_neg, num_stages):
    os.system(f'opencv_traincascade -data {output_dir} -vec {positives_vec} -bg {negatives_txt} '
              f'-numPos {num_pos} -numNeg {num_neg} -numStages {num_stages} -w 24 -h 24')

if __name__ == "__main__":
    train_classifier('dataset/positives.vec', 'dataset/negatives.txt', 'haar-cascade', 900, 500, 10)
