#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Library imports
import numpy as np
import netCDF4 as nc
import pandas as pd
import sys
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371e3
    t1 = lat1 * (np.pi/180)
    t2 = lat2 * (np.pi/180)
    delta_t = (lat2 - lat1) * (np.pi/180)
    delta_l = (lon2 - lon1) * (np.pi/180)
    sin_dt_2 = np.sin(delta_t/2)
    sin_dl_2 = np.sin(delta_l/2)
    a = (sin_dt_2 * sin_dt_2) + (np.cos(t1) * np.cos(t2) * sin_dl_2 * sin_dl_2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c # in m
    return d
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
def indexs_regrid(lat_smos,lon_smos,lat_topaz,lon_topaz,lat_tsurf,lon_tsurf,lat_sic,lon_sic):
    csv_topaz = pd.DataFrame()
    csv_tsurf = pd.DataFrame()
    csv_sic = pd.DataFrame()
    indexs_topaz=[]
    for k in range(0,len(lon_topaz)):
        print(str(k)+"/"+str(len(lon_topaz)))
        dist = np.abs(haversine_distance(lat_smos,lon_smos,lat_topaz[k],lon_topaz[k]))
        indexs_topaz.append(np.argmin(dist))
    csv_topaz['myid_topaz'] = indexs_topaz
    indexs_tsurf=[]
    for k in range(0,len(lon_tsurf)):
        print(str(k)+"/"+str(len(lon_tsurf)))
        dist = np.abs(haversine_distance(lat_smos,lon_smos,lat_tsurf[k],lon_tsurf[k]))
        indexs_tsurf.append(np.argmin(dist))
    csv_tsurf['myid_tsurf'] = indexs_tsurf
    indexs_sic=[]
    for k in range(0,len(lon_sic)):
        print(str(k)+"/"+str(len(lon_sic)))
        dist = np.abs(haversine_distance(lat_smos,lon_smos,lat_sic[k],lon_sic[k]))
        indexs_sic.append(np.argmin(dist))
    csv_sic['myid_sic'] = indexs_sic
    return csv_topaz, csv_tsurf, csv_sic
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Data loading
file_topaz="/Users/ferran/Library/CloudStorage/GoogleDrive-ferran.hernandez@isardsat.cat/My Drive/ICM/SIC_SIT_product/SIT_retrieval/uls_validation/topaz_cmems/cmems_mod_arc_phy_my_topaz4_P1D-m_20160101_20160415.nc"
file_tsurf="/Users/ferran/Library/CloudStorage/GoogleDrive-ferran.hernandez@isardsat.cat/My Drive/ICM/SIC_SIT_product/SIT_retrieval/uls_validation/tsurf_cmems/cmems_obs_si_arc_phy_my_L4-DMIOI_P1D-m_20160101_20160415.nc"
file_smos="/Users/ferran/Library/CloudStorage/GoogleDrive-ferran.hernandez@isardsat.cat/My Drive/ICM/SIC_SIT_product/SIT_retrieval/uls_validation/smos_sit/20160101_20160415/SMOS_Icethickness_v3.3_north_20160101.nc"
file_sic="/Users/ferran/Library/CloudStorage/GoogleDrive-ferran.hernandez@isardsat.cat/My Drive/ICM/SIC_SIT_product/SIT_retrieval/uls_validation/sic_osisaf/20160101_20160415/ice_conc_nh_polstere-100_multi_201601011200.nc"
ds_topaz=nc.Dataset(file_topaz)
ds_tsurf=nc.Dataset(file_tsurf)
ds_smos=nc.Dataset(file_smos)
ds_sic=nc.Dataset(file_sic)
lat_topaz=np.reshape(np.array(ds_topaz['latitude'][:]),-1)
lon_topaz=np.reshape(np.array(ds_topaz['longitude'][:]),-1)
lat_tsurf=np.array(ds_tsurf['lat'][:])
lon_tsurf=np.array(ds_tsurf['lon'][:])
lon_tsurf,lat_tsurf=np.meshgrid(lon_tsurf,lat_tsurf)
lat_tsurf=np.reshape(lat_tsurf,-1)
lon_tsurf=np.reshape(lon_tsurf,-1)
lat_smos=np.reshape(np.array(ds_smos['latitude'][:]),-1)
lon_smos=np.reshape(np.array(ds_smos['longitude'][:]),-1)
lat_sic=np.reshape(np.array(ds_sic['lat'][:]),-1)
lon_sic=np.reshape(np.array(ds_sic['lon'][:]),-1)

csv_topaz = pd.DataFrame()
csv_tsurf = pd.DataFrame()
csv_sic = pd.DataFrame()
"""  
indexs_topaz=[]
for k in range(0,len(lon_topaz)):
    print(str(k)+"/"+str(len(lon_topaz)))
    dist = np.abs(haversine_distance(lat_smos,lon_smos,lat_topaz[k],lon_topaz[k]))
    indexs_topaz.append(np.argmin(dist))
csv_topaz['myid_topaz'] = indexs_topaz
indexs_tsurf=[]
for k in range(0,len(lon_tsurf)):
    print(str(k)+"/"+str(len(lon_tsurf)))
    dist = np.abs(haversine_distance(lat_smos,lon_smos,lat_tsurf[k],lon_tsurf[k]))
    indexs_tsurf.append(np.argmin(dist))
csv_tsurf['myid_tsurf'] = indexs_tsurf
indexs_sic=[]
for k in range(0,len(lon_sic)):
    print(str(k)+"/"+str(len(lon_sic)))
    dist = np.abs(haversine_distance(lat_smos,lon_smos,lat_sic[k],lon_sic[k]))
    indexs_sic.append(np.argmin(dist))
csv_sic['myid_sic'] = indexs_sic
"""   

indexs_topaz = np.load('/Users/ferran/Library/CloudStorage/GoogleDrive-ferran.hernandez@isardsat.cat/My Drive/phd/ml_sit/scripts/indexs/indexs_topaz.npy')
indexs_tsurf = np.load('/Users/ferran/Library/CloudStorage/GoogleDrive-ferran.hernandez@isardsat.cat/My Drive/phd/ml_sit/scripts/indexs/indexs_tsurf.npy') 
indexs_sic = np.load('/Users/ferran/Library/CloudStorage/GoogleDrive-ferran.hernandez@isardsat.cat/My Drive/phd/ml_sit/scripts/indexs/indexs_sic.npy')
csv_topaz['myid_topaz'] = indexs_topaz
csv_tsurf['myid_tsurf'] = indexs_tsurf
csv_sic['myid_sic'] = indexs_sic
csv_topaz.to_csv('/Users/ferran/Library/CloudStorage/GoogleDrive-ferran.hernandez@isardsat.cat/My Drive/phd/ml_sit/scripts/indexs/indexs_topaz.csv')
csv_tsurf.to_csv('/Users/ferran/Library/CloudStorage/GoogleDrive-ferran.hernandez@isardsat.cat/My Drive/phd/ml_sit/scripts/indexs/indexs_tsurf.csv')
csv_sic.to_csv('/Users/ferran/Library/CloudStorage/GoogleDrive-ferran.hernandez@isardsat.cat/My Drive/phd/ml_sit/scripts/indexs/indexs_sic.csv')
csv = pd.concat([csv_topaz,csv_tsurf,csv_sic])
csv.to_csv('/Users/ferran/Library/CloudStorage/GoogleDrive-ferran.hernandez@isardsat.cat/My Drive/phd/ml_sit/scripts/indexs/indexs.csv')

#---------------------------------------------------------------------------------------------------------------------------------------------------------------#