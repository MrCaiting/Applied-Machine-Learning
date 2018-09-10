# The script for CS 298 AML MP1 Part 2
import pandas as pd
import numpy as np
from skimage.transform import resize
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

df_tr = pd.read_csv('train.csv')
df_te = pd.read_csv('test.csv')
df_val = pd.read_csv('val.csv')

# Read the column names
# df.columns
df_tr = df_tr.drop(columns='Unnamed: 0')    # Only Training data needs this

# Gettomg ready with training/validating/testing data
train_df = df_tr.drop(columns='label')
val_df = df_val.drop(columns='label')
train_labels = df_tr['label'].values
val_labels = df_val['label'].values

train_untouched = train_df.values
val_untouched = val_df.values
test_untouched = df_te.values

def bounding_scaling(pdframe: pd.DataFrame):
    temp_pix = pdframe.values.reshape((28, 28))
    location = np.where(temp_pix != 0)
    # Getting Boundaries
    t_top = np.min(location[0])
    t_bottom = np.max(location[0])
    t_left = np.min(location[1])
    t_right = np.max(location[1])

    cropped_img = temp_pix[t_top: t_bottom+1, t_left: t_right+1]
    return resize(cropped_img, (20, 20), preserve_range=True).reshape((20*20,))

def get_prior(labels: np.ndarray):
    num_labels = labels.shape[0]
    prior_list = []
    prior_list.append((np.where(labels == 0)[0].shape[0]*1.) / num_labels)
    prior_list.append((np.where(labels == 1)[0].shape[0]*1.) / num_labels)
    prior_list.append((np.where(labels == 2)[0].shape[0]*1.) / num_labels)
    prior_list.append((np.where(labels == 3)[0].shape[0]*1.) / num_labels)
    prior_list.append((np.where(labels == 4)[0].shape[0]*1.) / num_labels)
    prior_list.append((np.where(labels == 5)[0].shape[0]*1.) / num_labels)
    prior_list.append((np.where(labels == 6)[0].shape[0]*1.) / num_labels)
    prior_list.append((np.where(labels == 7)[0].shape[0]*1.) / num_labels)
    prior_list.append((np.where(labels == 8)[0].shape[0]*1.) / num_labels)
    prior_list.append((np.where(labels == 9)[0].shape[0]*1.) / num_labels)
    return np.array(prior_list)

train_prior = get_prior(train_labels)

train_df['rescaled'] = train_df.apply(bounding_scaling, axis=1)
val_df['rescaled'] = val_df.apply(bounding_scaling, axis=1)
df_te['rescaled'] = df_te.apply(bounding_scaling, axis=1)


train_scaled = np.vstack(train_df['rescaled'].values)
val_scaled = np.vstack(val_df['rescaled'].values)
test_scaled = np.vstack(df_te['rescaled'].values)

# Normalize the data
# train_features = preprocessing.scale(train_features)
g_nb = GaussianNB()
b_nb = BernoulliNB()
g_nb.fit(train_untouched, train_labels)
b_nb.fit(train_untouched, train_labels)
