import os
import json

def findAllFile(path):
    for root, ds, fs in os.walk(path):
        for f in fs:
            if f.endswith('.json'):
                yield f[:-5]

label_dict = dict()
label_num = 0

base = r"D:\Desktop\data final\data final"
json_dir = os.path.join(base, "json")
labels_dir = os.path.join(base, "labels")

print('Looking for JSON files...')
if not os.path.exists(json_dir):
    print('Error: JSON files directory not found:', json_dir)
else:
    print('JSON files directory found.')

    if not os.path.exists(labels_dir):
        print('Creating labels directory...')
        os.makedirs(labels_dir)

    for name in findAllFile(json_dir):
        print('Creating TXT file for', name)
        with open(os.path.join(base, "json", f"{name}.json"), 'r') as jf:
            data = json.load(jf)
        height = data["imageHeight"]
        width = data["imageWidth"]

        s = ""
        for item in data["shapes"]:
            label = item["label"]
            if label not in label_dict.keys():
                label_dict[label] = label_num
                label_num += 1

            s += str(label_dict[label]) + " "
            
            points = item["points"]
            for point in points:
                s += str(point[0] / width) + " "
                s += str(point[1] / height) + " "
            s = s[:-1] + "\n"

        with open(os.path.join(labels_dir, f"{name}.txt"), 'w') as tf:
            tf.write(s)
            

    yaml_path = os.path.join(base, "dataset.yaml")
    with open(yaml_path, "w") as f:
        print('Creating YAML file...')
        f.write(f"path: {base}\n")
        f.write("train: images\n")
        f.write("val: images\n")
        f.write("test: \n\n")
        f.write("names:\n")
        for key, num in label_dict.items():
            f.write(f"  {num}:{key}\n")
        
    print("Process completed.")
