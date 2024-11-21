#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Library imports
import numpy as np
import netCDF4 as nc
import pandas as pd
import time
import os
import sys
sys.path.append("/Users/ferran/Library/CloudStorage/GoogleDrive-ferran.hernandez@isardsat.cat/My Drive/phd/seaice_emission")
from burke_model import d_retrieved_burke_solver # type: ignore
from sice_empirical import sice_empirical # type: ignore
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Map generation
def map_generation(period):
    start_time = time.time()
    #file_topaz="/Users/ferran/Library/CloudStorage/GoogleDrive-ferran.hernandez@isardsat.cat/My Drive/ICM/SIC_SIT_product/SIT_retrieval/uls_validation/topaz_cmems/cmems_mod_arc_phy_my_topaz4_P1D-m_"+str(period)+".nc"
    file_tsurf="/Users/ferran/Library/CloudStorage/GoogleDrive-ferran.hernandez@isardsat.cat/My Drive/ICM/SIC_SIT_product/SIT_retrieval/uls_validation/tsurf_cmems/cmems_obs_si_arc_phy_my_L4-DMIOI_P1D-m_"+str(period)+".nc"
    files_topaz=np.sort(os.listdir("/Users/ferran/Library/CloudStorage/GoogleDrive-ferran.hernandez@isardsat.cat/My Drive/ICM/SIC_SIT_product/SIT_retrieval/uls_validation/topaz_cmems/"+str(period)+"/"))    
    files_smos=np.sort(os.listdir("/Users/ferran/Library/CloudStorage/GoogleDrive-ferran.hernandez@isardsat.cat/My Drive/ICM/SIC_SIT_product/SIT_retrieval/uls_validation/smos_sit/"+str(period)+"/"))
    files_sic=np.sort(os.listdir("/Users/ferran/Library/CloudStorage/GoogleDrive-ferran.hernandez@isardsat.cat/My Drive/ICM/SIC_SIT_product/SIT_retrieval/uls_validation/sic_osisaf/"+str(period)+"/"))
    #files_smos=np.sort(os.listdir("/Users/ferran/Library/CloudStorage/GoogleDrive-ferran.hernandez@isardsat.cat/My Drive/phd/tb_maps_arctic_bec/2014/regrid/smos/"))
    #files_sic=np.sort(os.listdir("/Users/ferran/Library/CloudStorage/GoogleDrive-ferran.hernandez@isardsat.cat/My Drive/phd/tb_maps_arctic_bec/2014/regrid/sic/"))
    c = 0
    for k in range(0,len(files_smos)):
        #print(str(files_smos[k]))
        print(str(k)+"/"+str(len(files_smos)))
        #ds_topaz=nc.Dataset(file_topaz)
        ds_tsurf=nc.Dataset(file_tsurf)
        ds_topaz=nc.Dataset("/Users/ferran/Library/CloudStorage/GoogleDrive-ferran.hernandez@isardsat.cat/My Drive/ICM/SIC_SIT_product/SIT_retrieval/uls_validation/topaz_cmems/"+str(period)+"/"+str(files_topaz[k]))
        ds_smos=nc.Dataset("/Users/ferran/Library/CloudStorage/GoogleDrive-ferran.hernandez@isardsat.cat/My Drive/ICM/SIC_SIT_product/SIT_retrieval/uls_validation/smos_sit/"+str(period)+"/"+str(files_smos[k]))
        ds_sic=nc.Dataset("/Users/ferran/Library/CloudStorage/GoogleDrive-ferran.hernandez@isardsat.cat/My Drive/ICM/SIC_SIT_product/SIT_retrieval/uls_validation/sic_osisaf/"+str(period)+"/"+str(files_sic[k]))
        #ds_smos=nc.Dataset("/Users/ferran/Library/CloudStorage/GoogleDrive-ferran.hernandez@isardsat.cat/My Drive/phd/tb_maps_arctic_bec/2014/regrid/smos/"+str(files_smos[c]))
        #ds_sic=nc.Dataset("/Users/ferran/Library/CloudStorage/GoogleDrive-ferran.hernandez@isardsat.cat/My Drive/phd/tb_maps_arctic_bec/2014/regrid/sic/"+str(files_sic[c]))

        lat_topaz=np.array(ds_topaz['latitude'][:])
        lon_topaz=np.array(ds_topaz['longitude'][:])
        lat_tsurf=np.array(ds_tsurf['latitude'][:])
        lon_tsurf=np.array(ds_tsurf['longitude'][:])
        lat_smos=np.array(ds_smos['latitude'][:])
        lon_smos=np.array(ds_smos['longitude'][:])
        lat_sic=np.array(ds_sic['lat'][:])
        lon_sic=np.array(ds_sic['lon'][:])

        #sss_topaz=np.array(ds_topaz['so'][k,0,:,:])
        #sit_topaz=np.array(ds_topaz['sithick'][k,:,:])
        #snow_topaz=np.array(ds_topaz['sisnthick'][k,:,:])
        sss_topaz=np.array(ds_topaz['so'][0,0,:,:])
        sit_topaz=np.array(ds_topaz['sithick'][0,:,:])
        snow_topaz=np.array(ds_topaz['sisnthick'][0,:,:])
        sic=np.array(ds_sic['ice_conc'][0,:,:])
        tsurf=np.array(ds_tsurf['analysed_st'][k,:,:])
        i_smos=np.array(ds_smos['TB'][0,:,:])
        #i_smos=(np.array(ds_smos['TB_H_meas'][0,:,:]) + np.array(ds_smos['TB_V_meas'][0,:,:])) / 2
        #sit_smos=np.array(ds_smos['sea_ice_thickness'][0,:,:])
        #situnc_smos=np.array(ds_smos['ice_thickness_uncertainty'][0,:,:])

        df_tsurf = pd.read_csv('/Users/ferran/Library/CloudStorage/GoogleDrive-ferran.hernandez@isardsat.cat/My Drive/phd/ml_sit/scripts/indexs/indexs_tsurf.csv',index_col=0)
        tsurf = np.reshape(tsurf, -1)
        df_tsurf['tsurf'] = tsurf
        i_smos0 = np.reshape(i_smos,-1)
        df_smos = pd.DataFrame()
        df_smos['i_smos'] = i_smos0
        df_smos['myid_tsurf'] = df_smos.index
        df_smos = df_smos.drop(columns=['i_smos'])
        df = pd.merge_asof(df_smos, df_tsurf.sort_values('myid_tsurf'), on=['myid_tsurf'])
        tsurf_rg = np.reshape(df['tsurf'].to_numpy(),(896,608))

        df_sic = pd.read_csv('/Users/ferran/Library/CloudStorage/GoogleDrive-ferran.hernandez@isardsat.cat/My Drive/phd/ml_sit/scripts/indexs/indexs_sic.csv',index_col=0)
        sic = np.reshape(sic, -1)
        df_sic['sic'] = sic
        i_smos0 = np.reshape(i_smos,-1)
        df_smos = pd.DataFrame()
        df_smos['i_smos'] = i_smos0
        df_smos['myid_sic'] = df_smos.index
        df_smos = df_smos.drop(columns=['i_smos'])
        df = pd.merge_asof(df_smos, df_sic.sort_values('myid_sic'), on=['myid_sic'])
        sic_rg = np.reshape(df['sic'].to_numpy(),(896,608))

        df_topaz = pd.read_csv('/Users/ferran/Library/CloudStorage/GoogleDrive-ferran.hernandez@isardsat.cat/My Drive/phd/ml_sit/scripts/indexs/indexs_topaz.csv',index_col=0)
        sss_topaz = np.reshape(sss_topaz, -1)
        df_topaz['sss_topaz'] = sss_topaz
        i_smos0 = np.reshape(i_smos,-1)
        df_smos = pd.DataFrame()
        df_smos['i_smos'] = i_smos0
        df_smos['myid_topaz'] = df_smos.index
        df_smos = df_smos.drop(columns=['i_smos'])
        df = pd.merge_asof(df_smos, df_topaz.sort_values('myid_topaz'), on=['myid_topaz'])
        sss_topaz_rg = np.reshape(df['sss_topaz'].to_numpy(),(896,608))
        
        df_topaz = pd.read_csv('/Users/ferran/Library/CloudStorage/GoogleDrive-ferran.hernandez@isardsat.cat/My Drive/phd/ml_sit/scripts/indexs/indexs_topaz.csv',index_col=0)
        sit_topaz = np.reshape(sit_topaz, -1)
        df_topaz['sit_topaz'] = sit_topaz
        i_smos0 = np.reshape(i_smos,-1)
        df_smos = pd.DataFrame()
        df_smos['i_smos'] = i_smos0
        df_smos['myid_topaz'] = df_smos.index
        df_smos = df_smos.drop(columns=['i_smos'])
        df = pd.merge_asof(df_smos, df_topaz.sort_values('myid_topaz'), on=['myid_topaz'])
        sit_topaz_rg = np.reshape(df['sit_topaz'].to_numpy(),(896,608))
        
        df_topaz = pd.read_csv('/Users/ferran/Library/CloudStorage/GoogleDrive-ferran.hernandez@isardsat.cat/My Drive/phd/ml_sit/scripts/indexs/indexs_topaz.csv',index_col=0)
        snow_topaz = np.reshape(snow_topaz, -1)
        df_topaz['snow_topaz'] = snow_topaz
        i_smos0 = np.reshape(i_smos,-1)
        df_smos = pd.DataFrame()
        df_smos['i_smos'] = i_smos0
        df_smos['myid_topaz'] = df_smos.index
        df_smos = df_smos.drop(columns=['i_smos'])
        df = pd.merge_asof(df_smos, df_topaz.sort_values('myid_topaz'), on=['myid_topaz'])
        snow_topaz_rg = np.reshape(df['snow_topaz'].to_numpy(),(896,608))

        mask_ssstopaz = np.logical_or(sit_topaz_rg < 0, np.isnan(sit_topaz_rg))
        sss_topaz_rg[mask_ssstopaz] = 33
        mask_sittopaz = np.logical_or(sit_topaz_rg < 0, np.isnan(sit_topaz_rg))
        sit_topaz_rg[mask_sittopaz] = 0.1
        mask_snowtopaz = np.logical_or(snow_topaz_rg < 0, np.isnan(snow_topaz_rg))
        snow_topaz_rg[mask_snowtopaz] = 0
        mask_tsurf = np.logical_or(tsurf_rg < 200, np.isnan(tsurf_rg))
        tsurf_rg[mask_tsurf] = 250
        condition_mask = (
            (i_smos >= 0) &
            #(sit_smos >= 0) &
            (sic_rg > 0) &
            ((lat_smos > 58) | ((lon_smos < -100) | (lon_smos > -70)))
        )
        sit_test = sit_topaz_rg[condition_mask]
        sice_test = sice_empirical(sss_topaz_rg[condition_mask], sit_topaz_rg[condition_mask])
        tice_test = ((tsurf_rg[condition_mask] - 273.15) - 1.8) / 2
        i_test = i_smos[condition_mask]
        #sit_obs = sit_smos[condition_mask]
        lat_test = lat_smos[condition_mask]
        lon_test = lon_smos[condition_mask]
        sss_test = sss_topaz_rg[condition_mask]
        snow_test = snow_topaz_rg[condition_mask]
        snp_test = np.where(snow_test > 0, 1, 0)
        #situnc_test = situnc_smos[condition_mask]
        sic_test = sic_rg[condition_mask]

        preds_invb = []
        for i in range(0,len(i_test)):
            preds_invb.append(d_retrieved_burke_solver(0, i_test[i], 40, tice_test[i], sice_test[i], snp_test[i], 'vant', 0.25))
            #epre, epim = e_eff_mix(40, 10, 0, 2, sice_test[i], tice_test[i])
            #preds_invb.append(d_retrieved_burke_solver_ep(0, i_test[i], 40, tice_test[i], sice_test[i], snp_test[i], epre, epim))
        input_map=np.zeros((5,896,608))
        """
        for i in range(0,896):
            for j in range(0,608):
                lon_match = np.where(lon_test == lon_smos[i][j])
                lat_match = np.where(lat_test == lat_smos[i][j])
                common_indices = np.intersect1d(lon_match, lat_match)
                if common_indices.size > 0:
                    input_map[0][i][j] = preds_invb[common_indices[0]]
                    input_map[1][i][j] = i_test[common_indices[0]]
                    input_map[2][i][j] = tice_test[common_indices[0]]
                    input_map[3][i][j] = sice_test[common_indices[0]]
                    input_map[4][i][j] = snp_test[common_indices[0]]
        """
        coord_dict0 = {}
        coord_dict1 = {}
        coord_dict2 = {}
        coord_dict3 = {}
        coord_dict4 = {}
        for l in range(len(lon_test)):
            coord_dict0[(lon_test[l], lat_test[l])] = preds_invb[l]
            coord_dict1[(lon_test[l], lat_test[l])] = i_test[l]
            coord_dict2[(lon_test[l], lat_test[l])] = tice_test[l]
            coord_dict3[(lon_test[l], lat_test[l])] = sice_test[l]
            coord_dict4[(lon_test[l], lat_test[l])] = snp_test[l]
        for i in range(0, 896):
            for j in range(0, 608):
                coord = (lon_smos[i][j], lat_smos[i][j])
                if coord in coord_dict0:
                    input_map[0][i][j] = coord_dict0[coord]
                if coord in coord_dict1:
                    input_map[1][i][j] = coord_dict1[coord]        
                if coord in coord_dict2:
                    input_map[2][i][j] = coord_dict2[coord]        
                if coord in coord_dict3:
                    input_map[3][i][j] = coord_dict3[coord]        
                if coord in coord_dict4:
                    input_map[4][i][j] = coord_dict4[coord]   
        np.save('/Users/ferran/Library/CloudStorage/GoogleDrive-ferran.hernandez@isardsat.cat/My Drive/phd/ml_sit/input_maps/input_maps_vant/input_'+str(files_smos[k][29:37])+'.npy',input_map)
        #np.save('/Users/ferran/Library/CloudStorage/GoogleDrive-ferran.hernandez@isardsat.cat/My Drive/phd/tb_maps_arctic_bec/2014/maps/input_'+str(files_smos[c])+'.npy',input_map)
        print("--- %s seconds ---" % (time.time() - start_time))
        c += 1
    return
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#