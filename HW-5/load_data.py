"""
This script is used to help laod all txt across different folders
"""

import os
import pandas as pd

def list_files(input_dir):
    subdir_list:list[str]= os.listdir(input_dir)
    return [input_dir + os.path.sep + x for x in subdir_list]

def load_data(main_directory):
    subdir_list:list[str]= os.listdir(main_directory)
    modified_list = []
    data_dict = {}

    # Removing some unnecessary data paths
    for each_entry in subdir_list:
        if not (each_entry.endswith('MODEL') or each_entry.endswith('.txt') or each_entry.endswith('.m')):
            modified_list.append(main_directory+each_entry)

    # Start loading values
    for each_dir in modified_list:
        activity_name = os.path.basename(each_dir)
        data_dict[activity_name] = []

        # Iterate through all file within the current folder
        for each_file in list_files(each_dir):
            curr_features = pd.read_csv(each_file, header=None, sep=' ')
            data_dict[activity_name].append(curr_features)

    return data_dict
