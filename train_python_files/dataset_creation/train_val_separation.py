import os
import shutil
import random

def split_dataset(image_folder_path, label_folder_path, main_folder_path, train_ratio=0.8):
    # Define paths for train and val folders
    train_folder = os.path.join(main_folder_path, 'train')
    val_folder = os.path.join(main_folder_path, 'val')

    train_image_folder = os.path.join(train_folder, 'images')
    val_image_folder = os.path.join(val_folder, 'images')
    train_label_folder = os.path.join(train_folder, 'labels')
    val_label_folder = os.path.join(val_folder, 'labels')

    # Create train and val folders if they don't exist
    os.makedirs(train_image_folder, exist_ok=True)
    os.makedirs(val_image_folder, exist_ok=True)
    os.makedirs(train_label_folder, exist_ok=True)
    os.makedirs(val_label_folder, exist_ok=True)

    # Get all image files from the original image folder
    all_images = [f for f in os.listdir(image_folder_path) if os.path.isfile(os.path.join(image_folder_path, f))]
    
    # Print the files found for debugging
    print(f"Found {len(all_images)} image files.")

    # Filter out files that don't have a corresponding label file
    all_labels = [f for f in os.listdir(label_folder_path) if os.path.isfile(os.path.join(label_folder_path, f))]
    all_images = [f for f in all_images if os.path.splitext(f)[0] in [os.path.splitext(l)[0] for l in all_labels]]
    
    # Print the files that have corresponding labels for debugging
    print(f"{len(all_images)} images have corresponding label files.")

    # Shuffle the files
    random.shuffle(all_images)

    # Calculate the split index
    split_index = int(len(all_images) * train_ratio)

    # Split the files into train and val
    train_images = all_images[:split_index]
    val_images = all_images[split_index:]

    # Move image and corresponding label files to train and val folders
    for f in train_images:
        shutil.move(os.path.join(image_folder_path, f), os.path.join(train_image_folder, f))
        label_file = os.path.splitext(f)[0] + '.txt'
        shutil.move(os.path.join(label_folder_path, label_file), os.path.join(train_label_folder, label_file))

    for f in val_images:
        shutil.move(os.path.join(image_folder_path, f), os.path.join(val_image_folder, f))
        label_file = os.path.splitext(f)[0] + '.txt'
        shutil.move(os.path.join(label_folder_path, label_file), os.path.join(val_label_folder, label_file))

    print(f"Dataset split: {len(train_images)} files in train, {len(val_images)} files in val.")

# Example usage
image_folder_path = r"D:\Desktop\data final\data final\images"  # Change this to your image folder path
label_folder_path = r"D:\Desktop\data final\data final\labels"  # Change this to your label folder path
main_folder_path = r'D:\Desktop\ultralytics\ultralytics\mydata\data_final'  # Change this to your main folder path
split_dataset(image_folder_path, label_folder_path, main_folder_path)
