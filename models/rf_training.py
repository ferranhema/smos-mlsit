#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Library imports
import numpy as np
import os
import netCDF4 as nc
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
def get_training_data(maps_path_1, maps_path_2):
    # Load input maps
    maps_dir = np.sort(os.listdir(str(maps_path_1)))
    dices0 = []
    tbs0 = []
    tices0 = []
    sices0 = []
    snps0 = []
    for map in maps_dir:
        print(map)
        input_map = np.load(str(maps_path_1) + '/' + str(map))
        dices0.append(np.ravel(input_map[0,:,:]))
        tbs0.append(np.ravel(input_map[1,:,:]))
        tices0.append(np.ravel(input_map[2,:,:]))
        sices0.append(np.ravel(input_map[3,:,:]))
        snps0.append(np.ravel(input_map[4,:,:]))
    maps_dir = np.sort(os.listdir(str(maps_path_2)))
    dices1 = []
    tbs1 = []
    tices1 = []
    sices1 = []
    snps1 = []
    for map in maps_dir:
        print(map)
        input_map = np.load(str(maps_path_2)+ '/' + str(map))
        dices1.append(np.ravel(input_map[0,:,:]))
        tbs1.append(np.ravel(input_map[1,:,:]))
        tices1.append(np.ravel(input_map[2,:,:]))
        sices1.append(np.ravel(input_map[3,:,:]))
        snps1.append(np.ravel(input_map[4,:,:]))
    dices = np.concatenate((np.ravel(dices0), np.ravel(dices1)))
    tbs = np.concatenate((np.ravel(tbs0), np.ravel(tbs1)))
    tices = np.concatenate((np.ravel(tices0), np.ravel(tices1)))
    sices = np.concatenate((np.ravel(sices0), np.ravel(sices1)))
    snps = np.concatenate((np.ravel(snps0), np.ravel(snps1)))
    
    # Filter out invalid values
    dice_train = dices[tbs != -999]
    tice_train = tices[tbs != -999]
    sice_train = sices[tbs != -999]
    snp_train = snps[tbs != -999]
    i_train = tbs[tbs != -999]

    # Gather training data
    input_train = np.concatenate((np.expand_dims(i_train, axis=1), np.expand_dims(tice_train, axis=1),
                                np.expand_dims(sice_train, axis=1), np.expand_dims(snp_train, axis=1)), axis=1)

    # Build training data frame
    data_train_burke = pd.DataFrame(data=input_train, columns=['intensity', 'temperature', 'salinity', 'snow presence'])
    data_train_burke['thickness'] = dice_train
    data_train_burke = data_train_burke.drop_duplicates(subset=['intensity'])
    data_train_burke = data_train_burke.drop_duplicates(subset=['temperature'])
    data_train_burke = data_train_burke.drop_duplicates(subset=['salinity'])

    # Extract a selected subset of the data for training
    data_train_burke = data_train_burke.iloc[100000:200000,:]

    return data_train_burke
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
def rf_train(maps_path_1, maps_path_2):
    # Load training data
    data_train_burke = get_training_data(maps_path_1, maps_path_2)

    # Training the RF algorithm
    rf = RandomForestRegressor(n_estimators=50, criterion='squared_error', random_state=42)
    rf.fit(data_train_burke.iloc[:,:4], data_train_burke.iloc[:,4])
    return rf
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#

maps_path_1 = '/Users/ferran/Library/CloudStorage/GoogleDrive-ferran.hernandez@isardsat.cat/My Drive/phd/ml_sit/input_maps/input_maps_vant/1920'
maps_path_2 = '/Users/ferran/Library/CloudStorage/GoogleDrive-ferran.hernandez@isardsat.cat/My Drive/phd/ml_sit/input_maps/input_maps_vant/2021'
rf_train(maps_path_1, maps_path_2)