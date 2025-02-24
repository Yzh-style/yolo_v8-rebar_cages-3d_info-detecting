# -*- coding: gbk -*-

import os

def rename_files(folder_path):
    # Define the mapping from the original names to the new names
    rename_map = {
        
        ' - 副本 (2)': '_1.3',
        ' - 副本 (3)': '_1.4',
        ' - 副本 (4)': '_1.2',
        ' - 副本 (5)': '_1.2.3',
        ' - 副本 (6)': '_1.2.4',
        ' - 副本 (7)': '_1.1',
        ' - 副本 (8)': '_1.1.3',
        ' - 副本 (9)': '_1.1.4',
        ' - 副本 (10)': '.3',
        ' - 副本 (11)': '.4',
        ' - 副本 (12)': '.2',
        ' - 副本 (13)': '.2.3',
        ' - 副本 (14)': '.2.4',
        ' - 副本 (15)': '.1',
        ' - 副本 (16)': '.1.3',
        ' - 副本 (17)': '.1.4',
        ' - 副本': '_1',
    }

    for filename in os.listdir(folder_path):
        new_filename = filename
        for old, new in rename_map.items():
            new_filename = new_filename.replace(old, new)
        if new_filename != filename:
            old_file = os.path.join(folder_path, filename)
            new_file = os.path.join(folder_path, new_filename)
            os.rename(old_file, new_file)
            print(f'Renamed: {filename} -> {new_filename}')

folder_path = r'D:\Desktop\data_column'
rename_files(folder_path)
