# Part 1 - Importing Libraries
# data analysis and wrangling
import pandas as pd

# importing modules
from data_preprocessing import feature_scaling_training, feature_scaling_testing, feature_creation_training, feature_creation_testing
from model_creation import create_model, train_model

# saving files and models
import pickle

# Part 2 - Loading Dataset
def loading_data():
    train_data = pd.read_csv("Dataset/Google_Stock_Price_Train.csv")
    test_data = pd.read_csv("Dataset/Google_Stock_Price_Test.csv")
    
    return train_data, test_data

# Part 3 - Data Preprocessing
def data_preprocessing(train_data, test_data, column_no, day_window):
    # Feature Scaling
    train_data_scaled, scaling = feature_scaling_training(train_data, column = column_no)
    # Creating Training Data
    X_train, Y_train = feature_creation_training(train_data_scaled, day_window=day_window)
    # Creating Test Data
    inputs, scaling = feature_scaling_testing(train_data, test_data, train_data.columns[1], scaling, day_window)
    X_test = feature_creation_testing(inputs, day_window)
    Y_test = test_data.iloc[:, 1:2].values

    return X_train, Y_train, X_test, Y_test, scaling

# Part 4 - Model Preparation
def model_creation(X_train):
    # Initialising RNN
    model = create_model(inputshape = (X_train.shape[1], 1))
    
    return model

# Part 5 - Model Training
def model_training(model, X_train, Y_train, no_of_epochs, batchsize):
    # Training Model
    model = train_model(model, X_train, Y_train, no_of_epochs, batchsize)
    
    return model

# Parameters
column_no = 1 #Open
day_window = 60
no_of_epochs = 10
batchsize = 32

train_data, test_data = loading_data()
X_train, Y_train, X_test, Y_test, scaling = data_preprocessing(train_data, test_data, column_no, day_window)
model = model_creation(X_train)
model_training(model, X_train, Y_train, no_of_epochs, batchsize)

# Saving model and Scaling the Open category
pickle.dump(model, 'LSTM.pkl')
pickle.sump(scaling, 'Scaling.pkl')
