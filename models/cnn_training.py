#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Library imports
import numpy as np
import matplotlib.pyplot as plt
import os
import keras
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
"""
Function that loads the training data for the CNN model
Inputs: path_maps - path to the maps
         input - list to store the input maps
         dices - list to store the output maps
Outputs: input - list with the input maps
         dices - list with the output maps
"""
def load_cnn_training_data(path_maps, input, dices):
    # Load input maps and extract the variables
    maps_dir = np.sort(os.listdir(path_maps))
    print(maps_dir)
    for map in maps_dir:
        print(map)
        input_map = np.load(str(path_maps) + str(map))
        arrays = [input_map[1,:,:], input_map[2,:,:], input_map[3,:,:], input_map[4,:,:]]
        input0 = np.stack(arrays, axis=2)
        input.append(input0)
        dices.append(np.stack([input_map[0,:,:]], axis=2))
    return input, dices
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
"""
Function that creates the CNN model
Inputs: None
Outputs: model - CNN model
"""
def cnn_model():
    # Model construction
    input_shape = (896, 608, 4)
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv2D(6, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(12, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(12, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(12, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(12, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(24, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(1, (3, 3), activation='linear', padding='same')
    ])
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mae'])    
    return model
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
"""
Function that trains the CNN model
Inputs: model - CNN model
        input_train - input maps for training
        output_train - output maps for training
        epochs - number of epochs
        batch_size - batch size
        save - boolean to save the model
        save_path - path to save the model
Outputs: None
"""
def cnn_train(model, input_train, output_train, epochs, batch_size, save, save_path):
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