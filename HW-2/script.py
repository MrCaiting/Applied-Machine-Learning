import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

# Set random seed so the result
np.random.seed(666)

train_df = pd.read_csv('train.data', header=None, na_values=' ?')
test_df = pd.read_csv('test.data', header=None, na_values=' ?')

# Used to visualize the NaN values
print('Training data with NaN across columns')
print(train_df.isnull().sum())
print('\nTesting data with NaN across columns')
print(test_df.isnull().sum())

def change_label(nonsense: str):
    """
    Helper function to change string labels into integers
    """
    if nonsense == ' <=50K':
        return -1
    elif nonsense == ' >50K':
        return 1
    else:
         return np.nan

train_df['label'] = train_df[14].apply(change_label)

# Dropping unnecessary columns
dropping_columns = [1, 3, 5, 6, 7, 8, 9, 13, 14]
train_df = train_df.drop(columns=dropping_columns)
test_df = test_df.drop(columns=dropping_columns[:len(dropping_columns)-1])

# Check NaN again
print('Training data with NaN across columns')
print(train_df.isnull().sum())
print('\nTesting data with NaN across columns')
print(test_df.isnull().sum())

# Remapping columns name
column_names = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
                'hours-per-week', 'label']
train_df.columns = column_names
test_df.columns = column_names[:len(column_names)-1]

# Exporting training and testing matrix
train_features = train_df.drop(columns='label').values
train_labels = train_df['label'].values
test_features = test_df.values

# Standardlization
train_standard = scale(train_features)
test_standard = scale(test_features)

def test_acc(feature: np.ndarray, label: np.ndarray, a: np.ndarray, b: int):
    predict = feature.dot(a) + b
    predict[predict > 0] = 1
    predict[predict <= 0] = -1
    if(label.shape[0] != predict.shape[0]):
        print('Something is wrong with the prediction array size.\n')
    result = predict + label
    acc = 1 - (1.*np.where(result == 0)[0].shape[0] / len(result))
    return acc, predict, result

def svm_sdg (epochs, lam, tol_steps, train_x, train_y, val_x, val_y):
    index_array = np.array(range(len(train_x)))

    a = np.random.random(train_x.shape[1])
    b = np.random.random(1)[0]

    for each in range(epochs):
        step_len = 1/(0.01*each + 50)
        # shuffle the entire data set at each epoch
        np.random.shuffle(index_array)
        hold_index = index_array[-50:]
        # split the data index array into a almost evenly splited array
        batch_index = np.array_split(index_array[:-50], tol_steps)

        for step in range(tol_steps):
            # Extract current minibatch data and labels
            x_k = train_x[batch_index[step], :]
            y_k = train_y[batch_index[step]]
            ax_b = x_k.dot(a) + b
            temp = y_k*ax_b
            big_idx = np.where(temp >= 1)[0]
            small_idx = np.where(temp < 1)[0]
            if len(big_idx) + len(small_idx) != len(temp):
                print("The length of separating indeces is wrong!!")
            delta_a = len(big_idx)*step_len*lam*a + step_len*(len(small_idx)*lam*a - y_k[small_idx].T.dot(x_k[small_idx, ]))
            delta_b = (-1)*step_len*np.sum(y_k[small_idx])
            # Updating a and b
            a -= 1./len(batch_index[step])*delta_a
            b -= 1./len(batch_index[step])*delta_b

            # if (step%30 == 0):
            #     curr_acc = test_acc(train_x[hold_index], train_y[hold_index], a, b)
            #     a_norm = np.linalg.norm(a)

        valid_acc, predict, result = test_acc(val_x, val_y, a, b)
        print('Validation accuracy: ', valid_acc)
    return predict, result

trainX, valX, trainY, valY = train_test_split(train_standard, train_labels, test_size=0.2, random_state=666)
svm_sdg(50, 1e-3, 300, trainX, trainY, valX, valY)
