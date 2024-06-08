import os
import src.prepare_data as prepare_data
import src.train_lbp as train_lbp
import src.detect_lbp as detect_lbp

if __name__ == "__main__":
    # Prepare data
    prepare_data.create_positives_file('dataset/positives', 'dataset/positives.txt')
    prepare_data.create_negatives_file('dataset/negatives', 'dataset/negatives.txt')

    # Train LBP classifier
    train_lbp.train_classifier('dataset/positives.txt', 'dataset/negatives.txt', 'lbp-cascade', 900, 500, 10)

    # Detect objects using the trained LBP classifier
    detect_lbp.detect_objects('test.jpg', 'lbp-cascade/cascade.xml')
