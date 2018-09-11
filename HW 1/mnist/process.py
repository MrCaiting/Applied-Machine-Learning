# The script for CS 298 AML MP1 Part 2
import pandas as pd
import numpy as np
from skimage.transform import resize
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier

df_tr = pd.read_csv('train.csv')
df_te = pd.read_csv('test.csv', header= None)
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

#######################################################
# Begin Naive Bayes Training
g_nb_1 = GaussianNB()
g_nb_2 = GaussianNB()
b_nb_1 = BernoulliNB()
b_nb_2 = BernoulliNB()

# For untouched
g_nb_1.fit(train_untouched, train_labels)
b_nb_1.fit(train_untouched, train_labels)
# For streched
g_nb_2.fit(train_scaled, train_labels)
b_nb_2.fit(train_scaled, train_labels)

# Validating
print('Gaussian + untouched validation acc:' , g_nb_1.score(val_untouched, val_labels))
print('Gaussian + stretched validation acc:' , g_nb_2.score(val_scaled, val_labels))
print('Bernoulli + untouched validation acc:' , b_nb_1.score(val_untouched, val_labels))
print('Bernoulli + stretched validation acc:' , b_nb_2.score(val_scaled, val_labels))

# Deleting models
del g_nb_1, g_nb_2, b_nb_1, b_nb_2

# Stack training data with validating data
total_untouched = np.vstack((train_untouched, val_untouched))
total_scaled = np.vstack((train_scaled, val_scaled))
total_labels = np.concatenate((train_labels, val_labels))

# re-do Naive Bayes Training
g_nb_1 = GaussianNB()
g_nb_2 = GaussianNB()
b_nb_1 = BernoulliNB()
b_nb_2 = BernoulliNB()

# For untouched
g_nb_1.fit(total_untouched, total_labels)
b_nb_1.fit(total_untouched, total_labels)
# For streched
g_nb_2.fit(total_scaled, total_labels)
b_nb_2.fit(total_scaled, total_labels)

# Getting prediction and save it into .csv file with correct column label
# @NOTE: I am lazy so I won't do anything fancy
prediction_1 = g_nb_1.predict(test_untouched)
pred_1_df = pd.DataFrame(data={'Label': prediction_1})
pred_1_df.to_csv('cwu72_1.csv', index=True)
pred_1_df = pd.read_csv('cwu72_1.csv')
pred_1_df.columns = ['ImageId', 'Label']
pred_1_df.to_csv('cwu72_1.csv', index=False)

prediction_2 = g_nb_2.predict(test_scaled)
pred_2_df = pd.DataFrame(data={'Label': prediction_2})
pred_2_df.to_csv('cwu72_2.csv', index=True)
pred_2_df = pd.read_csv('cwu72_2.csv')
pred_2_df.columns = ['ImageId', 'Label']
pred_2_df.to_csv('cwu72_2.csv', index=False)

prediction_3 = b_nb_1.predict(test_untouched)
pred_3_df = pd.DataFrame(data={'Label': prediction_3})
pred_3_df.to_csv('cwu72_3.csv', index=True)
pred_3_df = pd.read_csv('cwu72_3.csv')
pred_3_df.columns = ['ImageId', 'Label']
pred_3_df.to_csv('cwu72_3.csv', index=False)

prediction_4 = b_nb_2.predict(test_scaled)
pred_4_df = pd.DataFrame(data={'Label': prediction_4})
pred_4_df.to_csv('cwu72_4.csv', index=True)
pred_4_df = pd.read_csv('cwu72_4.csv')
pred_4_df.columns = ['ImageId', 'Label']
pred_4_df.to_csv('cwu72_4.csv', index=False)
del pred_1_df, pred_2_df, pred_3_df, pred_4_df

# Getting Mean Images
df_te['g1_pred'] = prediction_1
df_te['g2_pred'] = prediction_2
df_te['b1_pred'] = prediction_3
df_te['b2_pred'] = prediction_4


def get_mean_list (d_frame: pd.DataFrame, specify: str):
    num_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    mean_list = []
    if specify == 'g2_pred' or specify == 'b2_pred':
        for each in num_list:
            temp_array = df_te[df_te['specify'] == each]['rescaled'].values
            mean_list.append(np.mean(temp_array, axis=0))
    elif specify == 'g1_pred' or specify == 'b1_pred':
        for each in num_list:
            temp_df = df_te[df_te['specify'] == each]
            temp_df = temp_df.drop(columns = ['rescaled', 'g1_pred', 'g2_pred', 'b1_pred', 'b2_pred'])
            temp_array = temp_df.values
            mean_list.append(np.mean(temp_array, axis=0))
    else:
        print('Invalid String!')

    return mean_list

g1_list = get_mean_list(df_te, 'g1_pred')
g2_list = get_mean_list(df_te, 'g2_pred')
b1_list = get_mean_list(df_te, 'b1_pred')
b2_list = get_mean_list(df_te, 'b2_pred')


#######################################################
# Begin Random Forest Training
rfc_1 = RandomForestClassifier(n_estimators=10, max_depth=4)
rfc_2 = RandomForestClassifier(n_estimators=10, max_depth=4)
rfc_3 = RandomForestClassifier(n_estimators=10, max_depth=16)
rfc_4 = RandomForestClassifier(n_estimators=10, max_depth=16)
rfc_5 = RandomForestClassifier(n_estimators=30, max_depth=4)
rfc_6 = RandomForestClassifier(n_estimators=30, max_depth=4)
rfc_7 = RandomForestClassifier(n_estimators=30, max_depth=16)
rfc_8 = RandomForestClassifier(n_estimators=30, max_depth=16)

# Start Training
rfc_1.fit(total_untouched, total_labels)
rfc_2.fit(total_scaled, total_labels)
rfc_3.fit(total_untouched, total_labels)
rfc_4.fit(total_scaled, total_labels)
rfc_5.fit(total_untouched, total_labels)
rfc_6.fit(total_scaled, total_labels)
rfc_7.fit(total_untouched, total_labels)
rfc_8.fit(total_scaled, total_labels)

# Getting prediction and save it into .csv file with correct column label
# @NOTE: Same as before, still lazy
prediction = rfc_1.predict(test_untouched)
pred_df = pd.DataFrame(data={'Label': prediction})
pred_df.to_csv('cwu72_5.csv', index=True)
pred_df = pd.read_csv('cwu72_5.csv')
pred_df.columns = ['ImageId', 'Label']
pred_df.to_csv('cwu72_5.csv', index=False)

prediction = rfc_2.predict(test_scaled)
pred_df = pd.DataFrame(data={'Label': prediction})
pred_df.to_csv('cwu72_6.csv', index=True)
pred_df = pd.read_csv('cwu72_6.csv')
pred_df.columns = ['ImageId', 'Label']
pred_df.to_csv('cwu72_6.csv', index=False)

prediction = rfc_3.predict(test_untouched)
pred_df = pd.DataFrame(data={'Label': prediction})
pred_df.to_csv('cwu72_7.csv', index=True)
pred_df = pd.read_csv('cwu72_7.csv')
pred_df.columns = ['ImageId', 'Label']
pred_df.to_csv('cwu72_7.csv', index=False)

prediction = rfc_4.predict(test_scaled)
pred_df = pd.DataFrame(data={'Label': prediction})
pred_df.to_csv('cwu72_8.csv', index=True)
pred_df = pd.read_csv('cwu72_8.csv')
pred_df.columns = ['ImageId', 'Label']
pred_df.to_csv('cwu72_8.csv', index=False)

prediction = rfc_5.predict(test_untouched)
pred_df = pd.DataFrame(data={'Label': prediction})
pred_df.to_csv('cwu72_9.csv', index=True)
pred_df = pd.read_csv('cwu72_9.csv')
pred_df.columns = ['ImageId', 'Label']
pred_df.to_csv('cwu72_9.csv', index=False)

prediction = rfc_6.predict(test_scaled)
pred_df = pd.DataFrame(data={'Label': prediction})
pred_df.to_csv('cwu72_10.csv', index=True)
pred_df = pd.read_csv('cwu72_10.csv')
pred_df.columns = ['ImageId', 'Label']
pred_df.to_csv('cwu72_10.csv', index=False)

prediction = rfc_7.predict(test_untouched)
pred_df = pd.DataFrame(data={'Label': prediction})
pred_df.to_csv('cwu72_11.csv', index=True)
pred_df = pd.read_csv('cwu72_11.csv')
pred_df.columns = ['ImageId', 'Label']
pred_df.to_csv('cwu72_11.csv', index=False)

prediction = rfc_8.predict(test_scaled)
pred_df = pd.DataFrame(data={'Label': prediction})
pred_df.to_csv('cwu72_12.csv', index=True)
pred_df = pd.read_csv('cwu72_12.csv')
pred_df.columns = ['ImageId', 'Label']
pred_df.to_csv('cwu72_12.csv', index=False)

# Delete stuff gracefully
del prediction, pred_df
