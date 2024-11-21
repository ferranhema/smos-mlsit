#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Library imports
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.models import Sequential
from keras.layers import Dense, LSTM
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
"""
Function that loads the training data for the LSTM model
Inputs: path_maps - path to the maps
Outputs: data - input maps
"""
def load_lstm_training_data(path_maps):
    # Load input maps and extract the variables
    maps_dir = np.sort(os.listdir(path_maps))
    n_maps = maps_dir.shape[0]
    dices0 = np.zeros((n_maps,896*608))
    tbs0 = np.zeros((n_maps,896*608))
    tices0 = np.zeros((n_maps,896*608))
    sices0 = np.zeros((n_maps,896*608))
    snps0 = np.zeros((n_maps,896*608))
    c = 0
    for map in maps_dir:
        print(map)
        input_map = np.load(str(path_maps) + str(map))
        dices0[c,:] = np.ravel(input_map[0,:,:])
        tbs0[c,:] = np.ravel(input_map[1,:,:])
        tices0[c,:] = np.ravel(input_map[2,:,:])
        sices0[c,:] = np.ravel(input_map[3,:,:])
        snps0[c,:] = np.ravel(input_map[4,:,:])
        c += 1
    data0 = np.concatenate((np.reshape(tbs0,(tbs0.shape[0],tbs0.shape[1],1)),np.reshape(tices0,(tices0.shape[0],tices0.shape[1],1)),
                                np.reshape(sices0,(sices0.shape[0],sices0.shape[1],1)),np.reshape(snps0,(snps0.shape[0],snps0.shape[1],1)),
                                np.reshape(dices0,(dices0.shape[0],dices0.shape[1],1))), axis=-1)
    data = np.transpose(data0, (1, 0, 2))
    return data
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
"""
Function that prepares the data for the LSTM model
Inputs: data - input maps
Outputs: input - input maps for training
         output - output maps for training
"""
def data_preparation_lstm(data):
    # Shift the dataset to include previous day information
    input = data[:, 0 : data.shape[1] - 1, :]
    output = data[:, 1 : data.shape[1], -1]
    output = np.expand_dims(output, axis=-1)
    return input, output
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
"""
Function that filters the training data for the LSTM model
Inputs: input_train - input maps for training
        output_train - output maps for training
Outputs: input_train - filtered input maps for training
         output_train - filtered output maps for training
"""
def filter_lstm_training_data(input_train, output_train):
    # Create boolean masks for input_train and output_train
    mask_input = np.any(input_train != 0, axis=(1, 2))
    mask_output = np.any(output_train != 0, axis=(1, 2))
    # Combine the masks
    mask = mask_input & mask_output
    # Filter input_train and output_train using the mask
    input_train = input_train[mask]
    output_train = output_train[mask]
    return input_train, output_train
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
"""
Function that creates the LSTM model
Inputs: input_train - input maps for training
Outputs: model - LSTM model
"""
def lstm_model(input_train):
    # Model construction
    model = Sequential()
    model.add(LSTM(5, input_shape=(input_train.shape[1], input_train.shape[2]), activation='tanh', recurrent_activation='sigmoid', return_sequences=True))
    model.add(LSTM(10, activation='tanh', recurrent_activation='sigmoid', return_sequences=True))
    model.add(LSTM(5, activation='tanh', recurrent_activation='sigmoid', return_sequences=False))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mae', optimizer='adam')    
    return model
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
"""
Function that trains the LSTM model
Inputs: model - LSTM model
        input_train - input maps for training
        output_train - output maps for training
        epochs - number of epochs
        batch_size - batch size
        save - boolean to save the model
        save_path - path to save the model
Outputs: None
"""
def lstm_train(model, input_train, output_train, epochs, batch_size, save, save_path):
    # Train the model
    history = model.fit(input_train, output_train, epochs = epochs, batch_size = batch_size, verbose=2)
    if save == True:
        model.save(str(save_path))
    # Plot training history
    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.legend()
    plt.show()
    return
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#