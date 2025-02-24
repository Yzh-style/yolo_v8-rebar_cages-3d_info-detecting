import os
import json

def replace_in_json(file_path, old_str, new_str):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert JSON object to string
    json_str = json.dumps(data)
    
    # Replace the target string
    json_str = json_str.replace(old_str, new_str)
    
    # Convert string back to JSON object
    data = json.loads(json_str)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def process_folder(folder_path, old_str, new_str):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                replace_in_json(file_path, old_str, new_str)
                print(f"Processed file: {file_path}")

# Set the folder path and strings to be replaced
folder_path = "D:\Desktop\project"  
old_str = 'L-R-20'
new_str = 'L-R-6'

process_folder(folder_path, old_str, new_str)
