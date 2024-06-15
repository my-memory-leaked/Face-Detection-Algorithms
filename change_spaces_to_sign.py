import os

def replace_spaces_with_hyphens(directory):
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in dirs:
            new_name = name.replace(" ", "-")
            if new_name != name:
                old_path = os.path.join(root, name)
                new_path = os.path.join(root, new_name)
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed directory: {old_path} -> {new_path}")
                except Exception as e:
                    print(f"Error renaming directory {old_path} to {new_path}: {e}")
        
        for name in files:
            new_name = name.replace(" ", "-")
            if new_name != name:
                old_path = os.path.join(root, name)
                new_path = os.path.join(root, new_name)
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed file: {old_path} -> {new_path}")
                except Exception as e:
                    print(f"Error renaming file {old_path} to {new_path}: {e}")

if __name__ == "__main__":
    dataset_dir = 'dataset'  # Adjust this path to the correct dataset directory
    if os.path.exists(dataset_dir):
        replace_spaces_with_hyphens(dataset_dir)
    else:
        print(f"Directory {dataset_dir} does not exist.")
