import pickle
import numpy as np
import pandas as pd
import random

class SnufflerConvertor(object):
    def __init__(self, opt):
        self.opt = opt

    def __call__ (self, pick_df, catalog_df, catalogs, assignments):

        snuffler_file_name = './result/snuffler_output' + '/' + str(self.opt.time_interval[0]) + '_' + str(self.opt.time_interval[1]) + '_' + str(self.opt.time_interval[2]) + '_' + str(self.opt.time_interval[3]) + "_picks_snuffler.txt"

        # extract those picks which are removed by GaMMA
        removed_picks_gamma = pick_df[~pick_df.index.isin(assignments['pick_idx'])]

        with open(snuffler_file_name, "w") as f:
            f.write('# Snuffler Markers File Version 0.2\n')
        
            for j in range(len(removed_picks_gamma)):
                removed_pks = 'phase: ' + str(removed_picks_gamma["timestamp"].iloc[j]) + '  7 ' + str(removed_picks_gamma["id"].iloc[j]).upper()+ '.HHZ'+'    None           None       None         '+ str(removed_picks_gamma["type"].iloc[j]).upper() +'        None False'+ '\n'
                f.write(removed_pks)
                #phase: 2011-01-01 00:00:42.4360  0 CX.PB02..HHZ    None           None       None         P        None False


        # Group assignments
        grouped_assignments = assignments.groupby('event_idx')


        color_code = 0
        

        with open(snuffler_file_name, "a") as f:
            for event_idx, group_df in grouped_assignments:

                if color_code == 4:
                    color_code = 0

                #origin_time = catalogs[event_idx]['time']
                origin_time = catalog_df[catalog_df["event_index"] == event_idx].iloc[0]['time']
                intersection_indices = group_df['pick_idx'].tolist()
                extracted_picks = pick_df.loc[pick_df.index.isin(intersection_indices)]

                # event info
                event_code = 'C00000000000' + str(event_idx) + '_C1l5ZiDQwWCPA= '
                event_entry = 'event: ' + str(origin_time.split('T')[0]) + ' '+ str(origin_time.split('T')[1]) + '  5 ' + event_code + '-20.820710000000002 -70.15288000000002 None         None None  Event None'+ '\n'
                f.write(event_entry)
    
                for i in range(extracted_picks.shape[0]):
                    
                    pick_entry = 'phase: ' + str(extracted_picks["timestamp"].iloc[i]) + '  ' + str(color_code) + ' ' + str(extracted_picks["id"].iloc[i]).upper()+ '.HHZ'+ '    '+  event_code + str(origin_time.split('T')[0]) + '   '+ str(origin_time.split('T')[1]) +' ' + str(extracted_picks["type"].iloc[i]).upper() +'        None False'+ '\n'
                    #entry = 'phase: ' + str(picks["timestamp"].iloc[i]) + '  9 ' + str(picks["id"].iloc[i]).upper()+ ch+'    ' + event_code + 2017-01-01   01:27:00.1050 ' + str(picks["type"].iloc[i]).upper() +'        None False'+ '\n'
                    f.write(pick_entry)
            
                color_code += 1
            




        




    

    












































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