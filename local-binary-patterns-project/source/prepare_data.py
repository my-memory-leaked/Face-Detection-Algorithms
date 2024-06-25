import os

def create_positives_file(positives_dir, output_file):
    """
    Creates a file listing positive images with bounding box information.
    """
    with open(output_file, 'w') as f:
        for filename in os.listdir(positives_dir):
            if filename.endswith('.jpg'):
                f.write(f'{positives_dir}/{filename} 1 0 0 50 50\n')

def create_negatives_file(negatives_dir, output_file):
    """
    Creates a file listing negative images.
    """
    with open(output_file, 'w') as f:
        for filename in os.listdir(negatives_dir):
            if filename.endswith('.jpg'):
                f.write(f'{negatives_dir}/{filename}\n')

if __name__ == "__main__":
    create_positives_file('../dataset/lbp/positives', '../dataset/lbp/positives.txt')
    create_negatives_file('../dataset/lbp/negatives', '../dataset/lbp/negatives.txt')
