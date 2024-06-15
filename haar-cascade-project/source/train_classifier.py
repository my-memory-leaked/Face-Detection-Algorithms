import os
import struct
import subprocess
import shutil

def read_vec_file(vec_file):
    with open(vec_file, 'rb') as f:
        num_samples = struct.unpack('<i', f.read(4))[0]
        return num_samples

def verify_image_paths(file_path, base_dir):
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            image_path = os.path.abspath(os.path.join(base_dir, parts[0].replace('\\', '/')))
            if not os.path.exists(image_path):
                print(f"Image file {image_path} does not exist.")
                return False
    return True

def create_vec_file(modified_positives_txt, vec_file, num_samples, width=24, height=24):
    command = f'opencv_createsamples -info {modified_positives_txt} -num {num_samples} -w {width} -h {height} -vec {vec_file}'
    print(f'Running command: {command}')
    process = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(f'Command output: {process.stdout}')
    print(f'Command error: {process.stderr}')
    if process.returncode != 0:
        print(f"Error running createsamples command: {command}")
        return False
    return True

def train_classifier(positives_txt, negatives_txt, output_dir, num_pos, num_neg, num_stages, base_dir, width=24, height=24):
    base_dir = os.path.abspath(base_dir)
    positives_txt = os.path.abspath(positives_txt)
    negatives_txt = os.path.abspath(negatives_txt)
    output_dir = os.path.abspath(output_dir)
    
    print(f"Base Directory: {base_dir}")
    print(f"Positives File: {positives_txt}")
    print(f"Negatives File: {negatives_txt}")
    print(f"Output Directory: {output_dir}")

    if not os.path.exists(positives_txt):
        print(f"Positives file {positives_txt} does not exist.")
        return

    if not os.path.exists(negatives_txt):
        print(f"Negatives file {negatives_txt} does not exist.")
        return

    if not verify_image_paths(positives_txt, base_dir):
        print("Invalid paths in positives file.")
        return

    if not verify_image_paths(negatives_txt, base_dir):
        print("Invalid paths in negatives file.")
        return

    modified_positives_txt = 'modified_positives.txt'
    with open(positives_txt, 'r') as file, open(modified_positives_txt, 'w') as new_file:
        for line in file:
            parts = line.strip().split()
            image_path = os.path.abspath(os.path.join(base_dir, parts[0].replace('\\', '/')))
            new_file.write(f'{image_path} {" ".join(parts[1:])}\n')

    modified_negatives_txt = 'modified_negatives.txt'
    with open(negatives_txt, 'r') as file, open(modified_negatives_txt, 'w') as new_file:
        for line in file:
            image_path = os.path.abspath(os.path.join(base_dir, line.strip().replace('\\', '/')))
            new_file.write(f'{image_path}\n')

    vec_dir = os.path.join(base_dir, 'dataset/haar')
    os.makedirs(vec_dir, exist_ok=True)

    vec_file = os.path.join(vec_dir, 'positives.vec')
    
    if not create_vec_file(modified_positives_txt, vec_file, num_pos, width, height):
        print("Failed to create vec file.")
        return

    actual_num_pos = read_vec_file(vec_file)
    print(f"Number of Positive Samples in Vec File: {actual_num_pos}")

    # Remove existing output_dir if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    train_command = f'opencv_traincascade -data {output_dir} -vec {vec_file} -bg {modified_negatives_txt} ' \
                    f'-numPos {actual_num_pos} -numNeg {num_neg} -numStages {num_stages} -w {width} -h {height}'
    print(f'Running command: {train_command}')
    process = subprocess.run(train_command, shell=True, capture_output=True, text=True)
    print(f'Command output: {process.stdout}')
    print(f'Command error: {process.stderr}')
    if process.returncode != 0:
        print(f"Error running traincascade command: {train_command}")
        return

if __name__ == "__main__":
    positives_txt = 'D:/Repositories/wkiro-2024/dataset/haar/positives.txt'
    negatives_txt = 'D:/Repositories/wkiro-2024/dataset/haar/negatives.txt'
    output_dir = 'D:/Repositories/wkiro-2024/haar-cascade-project/haar-cascade'
    base_dir = 'D:/Repositories/wkiro-2024/'  # Adjust base_dir to the correct base path

    # User-defined number of positive and negative samples
    num_pos = 200  # Set your desired number of positive samples
    num_neg = 86  # Set your desired number of negative samples
    num_stages = 20  # Number of stages for training
    
    # Remove existing output_dir if it exists and create a new one
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    
    train_classifier(positives_txt, negatives_txt, output_dir, num_pos, num_neg, num_stages, base_dir)
