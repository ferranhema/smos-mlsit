#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Library imports
import numpy as np
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
"""
Function to calculate the sea ice salinity using an empirical formula
Inputs:
    Sw: Sea water salinity
    d_ice: Ice thickness
Outputs:
    sice: Sea ice salinity
"""
def sice_empirical(Sw,d_ice):
    import numpy as np
    a=0.5
    Sr=0.175
    return Sw*(1-Sr)*np.exp(-a*np.sqrt(d_ice*100))+Sr*Sw