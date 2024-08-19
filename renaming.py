# rename jpg files in a directory
# which is /perception_data/test9/rgb/

import os
import cv2
import argparse

def rename_files(base_dir):
    # Get the list of all files in directory tree at given path
    list_files = os.listdir(base_dir)
    print(list_files)
    # Iterate over all the entries
    for entry in list_files:
        # Create full path
        full_path = os.path.join(base_dir, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(full_path):
            rename_files(full_path)
        else:
            # rename the file to for example 1.jpg, 2.jpg, 3.jpg, etc not necessarily jpg files only change the name before the extension
            # there could also be npy files in the directory 
            if full_path.endswith('.jpg'):
                new_name = f'{base_dir}/{len(os.listdir(base_dir))}.jpg'
                os.rename(full_path, new_name)
            else:
                new_name = f'{base_dir}/{len(os.listdir(base_dir))}.npy'
                os.rename(full_path, new_name)

            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb_dir', type=str, default='perception_data/test9/rgb')
    parser.add_argument('--depth_dir', type=str, default='perception_data/test9/depth')
    args = parser.parse_args()
    rename_files(args.rgb_dir)
    rename_files(args.depth_dir)
    print("Press Enter to continue...")
            