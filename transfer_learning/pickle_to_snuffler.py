import pickle
import numpy as np
import pandas as pd
import random

def snuffler_convertor(file_name_p, file_name_s):
    # load p picks
    with open(file_name_p, 'rb') as fp:
        picker_p_picks = pickle.load(fp)    

    # load s picks
    with open(file_name_s, 'rb') as ft:
        picker_s_picks = pickle.load(ft)

    picks = pd.concat([picker_p_picks, picker_s_picks], axis=0)
    picks.sort_values(by=['timestamp'], inplace=True)
    with open("picks_snuffler.txt", "w") as f:
        f.write('# Snuffler Markers File Version 0.2\n')
        ch_list = ['.HHN', '.HHE']
        for i in range(picks.shape[0]):
            if picks["type"].iloc[i] == 'p':
                ch = '.HHZ'
            else:
                ch = random.choice(ch_list)
            
            entry = 'phase: ' + str(picks["timestamp"].iloc[i]) + '  3 ' + str(picks["id"].iloc[i]).upper()+ ch+'    F94dfYVHr0jq-KRD6eW73t57n24= 2017-01-01   01:27:00.1050 ' + str(picks["type"].iloc[i]).upper() +'        None False'+ '\n'
            f.write(entry)

def snuffler_convertor_ipoc(file_name):
    '''
    # load p picks
    with open(file_name_p, 'rb') as fp:
        picker_p_picks = pickle.load(fp)    

    # load s picks
    with open(file_name_s, 'rb') as ft:
        picker_s_picks = pickle.load(ft)

    picks = pd.concat([picker_p_picks, picker_s_picks], axis=0)
    '''
    with open(file_name, 'rb') as fp:
        picks = pickle.load(fp) 

    if 'timestamp' not in picks.columns:
        picks.rename(columns={'picks_time': 'timestamp'}, inplace=True)
        picks.rename(columns={'phase_hint': 'type'}, inplace=True)
        picks['type'] = picks['type'].str.lower()
        picks.drop(columns=["picks_uncertainty", "origins_time", "origins_longitude", "origins_latitude", "magnitudes"], inplace=True)
    #picks["id"] = picks["network_code"] + "." + picks["station_code"] + "."
    picks.sort_values(by=['timestamp'], inplace=True)
    with open(file_name.rsplit('/',1)[0] + "/" +file_name.rsplit('/',3)[-2]+"_"+"picks_snuffler.txt", "w") as f:
        f.write('# Snuffler Markers File Version 0.2\n')
        ch_list = ['.HHN', '.HHE']
        for i in range(picks.shape[0]):
            if picks["type"].iloc[i] == 'p':
                ch = '.HHZ'
            else:
                ch = random.choice(ch_list)
            
            entry = 'phase: ' + str(picks["timestamp"].iloc[i]) + '  9 ' + str(picks["id"].iloc[i]).upper()+ ch+'    F94dfYVHr0jq-KRD6eW73t57n24= 2017-01-01   01:27:00.1050 ' + str(picks["type"].iloc[i]).upper() +'        None False'+ '\n'
            f.write(entry)

if __name__ == "__main__":

    #file_name_p  = '/home/javak/Sample_data_chile/Comparing PhaseNet and Catalog/EQT_Instance/2019.355/PhaseNet_result_p_picks.pkl'
    #file_name_s = '/home/javak/Sample_data_chile/Comparing PhaseNet and Catalog/EQT_Instance/2019.355/PhaseNet_result_s_picks.pkl'

    file_name  = '/home/javak/Sample_data_chile/Comparing PhaseNet and Catalog/EQT_Instance_CJN/p=0.3_s=0.3_d=0.02/PhaseNet_result_p&s.pkl'

    snuffler_convertor_ipoc(file_name)