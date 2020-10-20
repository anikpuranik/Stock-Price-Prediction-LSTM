# importing libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def feature_scaling_training(dataset, column):
    scaling = MinMaxScaler(feature_range = (0, 1))
    data_scaled = dataset.iloc[:, column:column+1].values
    data_scaled = scaling.fit_transform(data_scaled)
    
    return data_scaled, scaling

def feature_scaling_testing(train_data, test_data, column_name, scaling, day_window):
    dataset_total = pd.concat((train_data[column_name], test_data[column_name]), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(test_data) - day_window:].values
    inputs = inputs.reshape(-1,1)
    inputs = scaling.transform(inputs)
    
    return inputs, scaling
    
def feature_creation_training(dataset, day_window):
    X = list()
    Y = list()
    for i in range(day_window, len(dataset)):
        X.append(dataset[i-day_window:i, 0])
        Y.append(dataset[i, 0])
    X, Y = np.array(X), np.array(Y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, Y

def feature_creation_testing(dataset, day_window):
    X = list()
    for i in range(day_window, len(dataset)):
        X.append(dataset[i-day_window:i, 0])
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X

