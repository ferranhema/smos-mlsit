import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import netCDF4 as nc
import pandas as pd
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import cmocean as cmo
import keras
import os
import imageio.v2 as imageio
from keras.models import Sequential
from keras.layers import Dense, LSTM
import sys
sys.path.append("/Users/ferran/Library/CloudStorage/GoogleDrive-ferran.hernandez@isardsat.cat/My Drive/phd/seaice_emission")
from sice_empirical import sice_empirical # type: ignore

#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
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
def data_preparation_lstm(data, initial_map, timesteps, features):
    # Shift the dataset to include previous day information
    input = data[:, 0 : data.shape[1] - 1, :]
    output = data[:, 1 : data.shape[1], -1]
    output = np.expand_dims(output, axis=-1)
    return input, output
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
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
def lstm_model():
    # Model construction
    model = Sequential()
    model.add(LSTM(5, input_shape=(input_train.shape[1], input_train.shape[2]), activation='tanh', recurrent_activation='sigmoid', return_sequences=True))
    model.add(LSTM(10, activation='tanh', recurrent_activation='sigmoid', return_sequences=True))
    model.add(LSTM(5, activation='tanh', recurrent_activation='sigmoid', return_sequences=False))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mae', optimizer='adam')    
    return model
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
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