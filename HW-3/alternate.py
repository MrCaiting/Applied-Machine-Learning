"""
    The script that contains all the necessary files
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
# from sklearn.metrics import mean_squared_error
from scipy.spatial import distance

component_list = [0, 1, 2, 3, 4]

df_1 = pd.read_csv('dataI.csv')
df_2 = pd.read_csv('dataII.csv')
df_3 = pd.read_csv('dataIII.csv')
df_4 = pd.read_csv('dataIV.csv')
df_5 = pd.read_csv('dataV.csv')
df_og = pd.read_csv('iris.csv')

data1 = df_1.values
data2 = df_2.values
data3 = df_3.values
data4 = df_4.values
data5 = df_5.values
og = df_og.values

all_data = {1: data1, 2: data2, 3: data3, 4: data4, 5: data5}
all_avg = {1: df_1.mean().values, 2: df_2.mean().values, 3: df_3.mean().values,
           4: df_4.mean().values, 5: df_5.mean().values}
og_avg = df_og.mean().values

def get_dist (arrayA, arrayB):
    summation = 0.
    for eachA, eachB in zip(arrayA, arrayB):
        summation += distance.sqeuclidean(eachA, eachB)
    return summation / len(arrayA)

def get_mse_list (data_set, og_data, data_avg, og_avg):
    n_list = []
    c_list = []
    for each in component_list:
        pca_c = PCA(n_components=each, svd_solver='full')
        pca_n = PCA(n_components=each, svd_solver='full')

        lower_c = pca_c.fit_transform(data_set)
        pca_n.fit(og_data)
        lower_n = pca_n.transform(data_set)
        result_c = pca_c.inverse_transform(lower_c)
        result_n = pca_n.inverse_transform(lower_n)

        # Calculating MSE using what is suggested by Piazza post
        n_list.append(get_dist(og_data, result_n))
        c_list.append(get_dist(og_data, result_c))
        del pca_c, pca_n
    return n_list + c_list

result_list = []
for key, data in all_data.items():
    result_list.append(get_mse_list(data, og, all_avg[key], og_avg))
df_num = pd.DataFrame(data=result_list)
df_num.columns = ['0N', '1N', '2N', '3N', '4N','0c', '1c', '2c', '3c', '4c']
df_num.to_csv('numbers.csv', index=False)

# Generating .csv file for the first task
pca = PCA(n_components = 2, svd_solver='full')
lower_d = pca.fit_transform(data2)
result_c = pca.inverse_transform(lower_d)
df = pd.DataFrame(data=result_c)
df.columns = ['X1', 'X2', 'X3', 'X4']
df.to_csv("recon.csv", index=False)
