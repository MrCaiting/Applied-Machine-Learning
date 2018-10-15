"""
The main file that is used to cluster and train the data
"""
import numpy as np
import pandas as pd
from load_data import load_data

data_path = 'HMP_Dataset/'

data_dict = load_data(data_path)

slice = 32

def cut_in_seg(raw_list, slice_size):
    """
    result_df: This should be the return dataframe with segmented data in desired format
    raw_list: This should be a list of the DataFrame in original shape
    segment_size: This is the segment sizing that we pass in to control how do we cut the array
    """
    result_list = []
    for each_df in raw_list:    # Iterate through every df in the list
        idx = 1        # Old school counter
        bigger_idx = 1
        curr_list_len = each_df.values.shape[0]
        temp = []
        for each_vec in each_df.values:
            temp.append(each_vec)
            if (idx == slice_size):
                result_list.append(np.asarray(temp))
                idx = 0     # Reset counter
                bigger_idx += 1
                temp = []
            if ((curr_list_len - bigger_idx*slice_size) < slice_size):   # Jump over all remaining item
                continue
            idx += 1
    return result_list
