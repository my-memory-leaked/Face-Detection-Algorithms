import os
import struct
import subprocess
import shutil
import cv2
from matplotlib import pyplot as plt

def read_vec_file(vec_file):
    with open(vec_file, 'rb') as f:
        num_samples = struct.unpack('<i', f.read(4))[0]
        return num_samples

def validate_vec_file(vec_file):
    with open(vec_file, 'rb') as f:
        num_samples = struct.unpack('<i', f.read(4))[0]
        sample_size = struct.calcsize('<iihh')
        bytes_left = os.path.getsize(vec_file) - 4
        actual_samples = bytes_left // sample_size
        
        if actual_samples != num_samples:
            print(f"Error: Expected {num_samples} samples, but found {actual_samples} in the vec file.")
        else:
            print(f"Vec file validated successfully with {actual_samples} samples.")
        return actual_samples

def verify_image_paths(file_path, base_dir):
    with open(file_path, 'r') as file:
        valid = True
        for line in file:
            parts = line.strip().split()
            image_path = os.path.abspath(os.path.join(base_dir, parts[0].replace('\\', '/')))
            if not os.path.exists(image_path):
                print(f"Image file {image_path} does not exist.")
                valid = False
            else:
                print(f"Verified {image_path}")
        return valid

def create_vec_file(modified_positives_txt, vec_file, num_samples, width=24, height=24):
    command = f'opencv_createsamples -info {modified_positives_txt} -num {num_samples} -w {width} -h {height} -vec {vec_file}'
    return run_command(command)

def run_command(command):
    print(f"Running command: {command}")
    process = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if process.stdout:
        print("Command output:", process.stdout.decode())
    if process.stderr:
        print("Command error:", process.stderr.decode())

    return process.returncode

def train_classifier(positives_txt, negatives_txt, output_dir, num_pos, num_neg, num_stages, base_dir, width=24, height=24):
    base_dir = os.path.abspath(base_dir)
    positives_txt = os.path.abspath(positives_txt)
    negatives_txt = os.path.abspath(negatives_txt)
    output_dir = os.path.abspath(output_dir)
    
    if not os.path.exists(positives_txt) or not os.path.exists(negatives_txt):
        print("Files missing: Positives or negatives files do not exist.")
        return

    if not verify_image_paths(positives_txt, base_dir) or not verify_image_paths(negatives_txt, base_dir):
        print("Invalid paths in input files.")
        return

    modified_positives_txt = 'modified_positives.txt'
    with open(positives_txt, 'r') as file, open(modified_positives_txt, 'w') as new_file:
        for line in file:
            parts = line.strip().split()
            image_path = parts[0].replace('\\', '/')
            # Correctly handle absolute and relative paths
            if not os.path.isabs(image_path):
                image_path = os.path.join(base_dir, image_path)
            image_path = os.path.abspath(image_path)
            new_file.write(f'{image_path} {" ".join(parts[1:])}\n')

    vec_dir = os.path.join(base_dir, 'dataset/haar')
    os.makedirs(vec_dir, exist_ok=True)

    vec_file = os.path.join(vec_dir, 'positives.vec')
    if create_vec_file(modified_positives_txt, vec_file, num_pos, width, height):
        actual_num_pos = validate_vec_file(vec_file)
        print(f"Number of Positive Samples in Vec File: {actual_num_pos}")

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        train_command = f'opencv_traincascade -data {output_dir} -vec {vec_file} -bg {negatives_txt} ' \
                        f'-numPos {actual_num_pos} -numNeg {num_neg} -numStages {num_stages} -w {width} -h {height}'
        if run_command(train_command) != 0:
            print(f"Error running traincascade command: {train_command}")


if __name__ == "__main__":
    base_dir = 'D:/Repositories/wkiro-2024/'  # Adjust base_dir to the correct base path
    train_classifier(
        positives_txt=os.path.join(base_dir, 'dataset/haar/positives.txt'),
        negatives_txt=os.path.join(base_dir, 'dataset/haar/negatives.txt'),
        output_dir=os.path.join(base_dir, 'haar-cascade-project/haar-cascade'),
        num_pos=2000,  # Set your desired number of positive samples
        num_neg=86,  # Set your desired number of negative samples
        num_stages=5,  # Number of stages for training
        base_dir=base_dir
    )
