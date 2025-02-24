import os
import random
from PIL import Image, ImageEnhance

def adjust_darker(input_folder, output_suffix=".2", factor=1.2):
    # Get all file names in a folder
    for filename in os.listdir(input_folder):
        # Check if the file is a .jpg image and does not contain '.'
        if filename.lower().endswith('.jpg') and '.1' not in filename[:-4]:  # Exclude the extension part for dot check
            # Opening an image file
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            random_number = round(random.uniform(-0.1, 0.1), 2)
            adjusted_factor = factor + random_number                              # introduce 20% randomness

            # Adjusting Brightness
            brightness_enhancer = ImageEnhance.Brightness(img)
            img_brightness_enhanced = brightness_enhancer.enhance(adjusted_factor)  # Reduce brightness by factor

            # Adjusting Contrast
            contrast_enhancer = ImageEnhance.Contrast(img_brightness_enhanced)
            img_enhanced = contrast_enhancer.enhance(adjusted_factor)  # Reduce contrast by factor

            # Constructing a new filename
            new_filename = f"{os.path.splitext(filename)[0]}{output_suffix}{os.path.splitext(filename)[1]}"
            new_img_path = os.path.join(input_folder, new_filename)
            print(f'adjust_darker: {filename} -> {new_filename}')
            
            # Saving the adjusted image
            img_enhanced.save(new_img_path)

def adjust_warmer(input_folder, output_suffix=".4", factor=0.9):
    # Get all file names in a folder
    for filename in os.listdir(input_folder):
        # Check if the file is a .jpg image and does not contain '.'
        if filename.lower().endswith('.jpg') and '.3' not in filename[:-4]:      # Exclude the extension part for dot check
            # Opening an image file
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            random_number = round(random.uniform(-0.05, 0.05), 3)
            adjusted_factor = factor + random_number                              # introduce 10% randomness
            # Adjusting Saturation
            color_enhancer = ImageEnhance.Color(img)
            img_color_enhanced = color_enhancer.enhance(factor)                   # Increase saturation by 20%

            # Adjusting Warmth by adjusting the image colors
            r, g, b = img_color_enhanced.split()
            r = r.point(lambda p: p * factor if p * factor <= 255 else 255)      # Increase red channel
            b = b.point(lambda p: p * (2 - factor) if p * (2 - factor) >= 0 else 0)  # Decrease blue channel
            img_warmer = Image.merge('RGB', (r, g, b))

            # Constructing a new filename
            new_filename = f"{os.path.splitext(filename)[0]}{output_suffix}{os.path.splitext(filename)[1]}"
            new_img_path = os.path.join(input_folder, new_filename)
            print(f'adjust_warmer: {filename} -> {new_filename}')

            # Saving the adjusted image
            img_warmer.save(new_img_path)
            
def delete_files_with_dot(input_folder):
    # Get all file names in a folder
    for filename in os.listdir(input_folder):
        # Check if the file name contains a dot
        if filename.lower().endswith('.json') and '_' in filename[:-5]:  # Exclude the extension part for dot check
            # Construct the full file path
            file_path = os.path.join(input_folder, filename)
            # Check if it is a file
            if os.path.isfile(file_path):
                print(f"Deleting file: {file_path}")
                os.remove(file_path)  # Delete the file


input_folder = 'D:\Desktop\data_column' 
#adjust_darker(input_folder)
#adjust_warmer(input_folder)
delete_files_with_dot(input_folder)