"""
The main file that is used to cluster and train the data
"""
import numpy as np
import pandas as pd
from load_data import load_data

data_path = 'HMP_Dataset/'

data_dict = load_data(data_path)

slice = 32
