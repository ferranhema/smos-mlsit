# SMOS-MLSIT
This repository contains the scripts and materials supporting the scientific paper: “Assessment of machine learning-driven retrievals of Arctic sea ice thickness from L-band radiometry remote sensing” by Ferran Hernández Macià et al. 2025.
The study presents an assessment of machine learning and deep learning retrievals to estimate thin sea ice thickness (SIT) from SMOS satellite data, while using ESA's official product as baseline.

## Repository Structure:
```
.
├── models/
│   ├── rf_training.py       # Code for training the Random Forest model
│   ├── lstm_training.py     # Code for training the LSTM model
│   ├── cnn_training.py      # Code for training the CNN model
│   └── main.py              # Script to manage the overall workflow
├── data_processing/         # Scripts for preparing and preprocessing data
│   ├── burke_model.py       # Code that contains the radiative transfer model from Burke et al., 1979 used to compute not only the brightess temperature, but also it is inverted to compute the sea ice thickness
│   ├── indexs_regrid.py     # Function to generate the indexs from co-locating the different data which are used to further regrid to a common grid
│   ├── map_generation.py    # Function to generate the maps that are used as input data tot train the models
│   ├── permittivity_models.py # Code that contains the functions to compute the permittivity/dielectric constant of the media components: snow, sea ice, sea water
│   └── sice_empirical.py    # Function to compute the sea ice salinity using an empirical formula
└── README.md                # This documentation file
```
