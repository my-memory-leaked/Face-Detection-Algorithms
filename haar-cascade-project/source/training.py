import os
import struct
import subprocess
import shutil

def run_command(command):
    print(f"Running command: {command}")
    process = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdout:
        print("Command output:", process.stdout.decode())
    if process.stderr:
        print("Command error:", process.stderr.decode())
    return process.returncode

def validate_vec_file(vec_file):
    with open(vec_file, 'rb') as f:
        header = f.read(12)  # Read the header to extract the number of samples
        if len(header) < 12:
            print("Invalid or corrupt vec file")
            return 0
        num_samples = struct.unpack('<i', header[8:12])[0]
        return num_samples

def create_vec_file(info_file, num, width, height, vec_file):
    command = f'opencv_createsamples -info {info_file} -num {num} -w {width} -h {height} -vec {vec_file}'
    return run_command(command)

def train_cascade(vec_file, bg_file, num_pos, num_neg, num_stages, data_dir, width, height):
    command = f'opencv_traincascade -data {data_dir} -vec {vec_file} -bg {bg_file} -numPos {num_pos} -numNeg {num_neg} -numStages {num_stages} -w {width} -h {height}'
    return run_command(command)

def prepare_paths(file_path, base_dir):
    with open(file_path, 'r') as file:
        new_content = []
        for line in file:
            parts = line.strip().split()
            image_path = parts[0].replace('\\', '/')
            if not os.path.isabs(image_path):
                image_path = os.path.join(base_dir, image_path)
            image_path = os.path.normpath(image_path)
            new_content.append(f'{image_path} {" ".join(parts[1:])}')
        return new_content

def write_prepared_paths(content, output_file):
    with open(output_file, 'w') as file:
        for line in content:
            file.write(line + '\n')

if __name__ == "__main__":
    base_dir = 'D:/Repositories/wkiro-2024/'  # Adjust base_dir to the correct base path
    positives_txt = os.path.join(base_dir, 'dataset/haar/positives.txt')
    negatives_txt = os.path.join(base_dir, 'dataset/haar/negatives.txt')
    vec_dir = os.path.join(base_dir, 'haar-cascade-project/haar-cascade')
    vec_file = os.path.join(vec_dir, 'positives.vec')
    output_dir = os.path.join(vec_dir, 'trained_cascade')

    modified_positives = positives_txt
    modified_positives_txt = 'modified_positives.txt'
    write_prepared_paths(modified_positives, modified_positives_txt)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    num_samples = 2000  # Adjust as necessary
    width = 24
    height = 24
    num_pos = int(num_samples * 0.85)  # Use 85% of samples
    num_neg = 1000  # Adjust based on your negatives dataset
    num_stages = 10  # Adjust based on desired complexity

    if create_vec_file(modified_positives_txt, num_samples, width, height, vec_file) == 0:
        if train_cascade(vec_file, negatives_txt, num_pos, num_neg, num_stages, output_dir, width, height) == 0:
            print("Training completed successfully!")
        else:
            print("Training failed.")
    else:
        print("Failed to create vector file.")
