#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Library imports
import numpy as np
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Function imports
from rf_training import rf_train
from cnn_training import load_cnn_training_data, cnn_model, cnn_train
from lstm_training import load_lstm_training_data, data_preparation_lstm, filter_lstm_training_data, lstm_model, lstm_train
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#

# Train the RF model
maps_path_1920 = '/Users/ferran/Library/CloudStorage/GoogleDrive-ferran.hernandez@isardsat.cat/My Drive/phd/ml_sit/input_maps/input_maps_vant/1920'
maps_path_2021 = '/Users/ferran/Library/CloudStorage/GoogleDrive-ferran.hernandez@isardsat.cat/My Drive/phd/ml_sit/input_maps/input_maps_vant/2021'
rf_train(maps_path_1920, maps_path_2021)

# Train the CNN model
input = []
dices = []
input, dices = load_cnn_training_data(maps_path_1920, input, dices)
input, dices = load_cnn_training_data(maps_path_2021, input, dices)
input = np.array(input)
output = np.array(dices)
X_train = input[:181,:,:,:]
X_test = input[181:,:,:,:]
y_train = output[:181,:,:,:]
y_test = output[181:,:,:,:]
model = cnn_model()
cnn_train(model, X_train, y_train, 200, 64, False, '')

# Train the LSTM model
data_train = load_lstm_training_data(maps_path_1920)
input_train, output_train = data_preparation_lstm(data_train, 18, 2, 5)
input_train, output_train = filter_lstm_training_data(input_train, output_train)
model = lstm_model(input_train)
lstm_train(model, input_train, output_train, 50, 2, True, '')