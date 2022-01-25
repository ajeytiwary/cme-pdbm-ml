import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
import os
from config import path, earth_cme_path, path_cme_soho, path_helcats, dbm_path
# import requests
# print(path)
# path = '/home/plasmion/CWI_analysis/CME_comparison/'

# earth_cme_path = path + 'cme_dat1/'
# earth_cme = 'table_geoeffective.csv'
def load_dataframes():
    CME_dat1 = pd.read_csv(earth_cme_path+ 'table_geoeffective.csv', header = [0], index_col= 1)


    # path_cme_soho = path + 'cme_dat2/'
    CME_dat2 = pd.read_csv(path_cme_soho + 'cme_filt_mass.csv', index_col = 0)

    # url = 'https://helioforecast.space/static/sync/icmecat/HELCATS_ICMECAT_v20.csv'
    # path_helcats = path + 'cme_dat3/'
    CME_dat3 = pd.read_csv(path_helcats + 'HELCATS_ICMECAT_v20.csv', index_col =0)
    # CME_dat3.to_csv('/home/plasmion/CWI_analysis/CME_comparision/cme_dat3/'+'HELCATS_v20.csv')

    # dbm_path = path + 'cme_dat4/'
    CME_dat4 = pd.read_csv(dbm_path + 'ICME_complete_dataset_v3.csv', header=[0,1])
    CME_dat1['LASCO_times'] = pd.to_datetime(CME_dat1[CME_dat1.columns[1]])
    CME_dat1['SOHO_times'] = CME_dat1['LASCO_times'] - \
        pd.Timedelta(hours = 1.5 )

    CME_dat2['LASCO_times'] = pd.to_datetime(CME_dat2['date'] +' '+ CME_dat2['time'])
    CME_dat2['SOHO_times'] = CME_dat2['LASCO_times'] - \
        pd.Timedelta(hours = 1.5 ) 

    CME_dat4['LASCO_times'] = pd.to_datetime(CME_dat4[CME_dat4.columns[1]])
    CME_dat4['SOHO_times'] = CME_dat4['LASCO_times'] - \
        pd.Timedelta(hours = 1.5)

    CME_dat1.name = 'Earth_directed_cmes'
    CME_dat2.name = 'SOHO_cmes'
    CME_dat3.name = 'HELCATS_cmes'
    CME_dat4.name = 'pdbm_cme'    



    return CME_dat1, CME_dat2, CME_dat3, CME_dat4 


def create_folders():
    dat_cme_earth_path = earth_cme_path +'data/'
    if not os.path.isdir(dat_cme_earth_path):
        os.makedirs(dat_cme_earth_path)

    dat_cme_earth_path = path_cme_soho +'data/'
    if not os.path.isdir(dat_cme_earth_path):
        os.makedirs(dat_cme_earth_path)      

    dat_cme_earth_path = path_helcats +'data/'
    if not os.path.isdir(dat_cme_earth_path):
        os.makedirs(dat_cme_earth_path)

    dat_cme_earth_path = dbm_path +'data/'
    if not os.path.isdir(dat_cme_earth_path):
        os.makedirs(dat_cme_earth_path)