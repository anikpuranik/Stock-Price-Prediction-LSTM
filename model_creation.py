from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

def create_model(inputshape):
    # Initialising model
    model = Sequential()
    
    # LSTM Layer 1
    model.add(LSTM(units=50, return_sequences=True, input_shape=inputshape))
    model.add(Dropout(0.2))

    # LSTM Layers 2
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    # LSTM Layers 3
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    
    # LSTM Layer 4 
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    
    # Output Layer
    model.add(Dense(units = 1))
    
    # Compiling RNN
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def train_model(model, train_data, test_data, epochs, batchsize):
    model.fit(train_data, test_data, epochs=epochs, batch_size=batchsize)
    
    return model
    
    
    