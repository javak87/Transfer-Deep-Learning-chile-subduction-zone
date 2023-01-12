import obspy
from typing import Tuple
import os
import pickle
import datetime
import seisbench
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt


class Picks_Comparison (object):

    def __init__(self, start_year_analysis, start_day_analysis, 
                    end_year_analysis, end_day_analysis, event_picks, Ground_truth):

        '''
        This class uses the start_year_analysis, start_day_analysis, end_year_analysis, and end_day_analysis, 
            and event_picks to filter phasenet picks and existing ground true. 
            At the end, picks belog to ground true has been considered as the base line.
            For each ground true picks, the closest pick in the phasenet picks has been identified and reported
            the distance.
        
        Parameters:
                - start_year_analysis = starting year of analysis
                - start_day_analysis = starting day of analysis
                - end_year_analysis = eding year of analysis
                - end_day_analysis = eding day of analysis
                - event_picks (data frame): Phasenet picks.
                    It should be noted that unassociated or associated picks can be used.
        '''
        self.start_year_analysis = start_year_analysis
        self.start_day_analysis = start_day_analysis
        self.end_year_analysis = end_year_analysis
        self.end_day_analysis = end_day_analysis
        self.event_picks = event_picks
        self.Ground_truth = Ground_truth
        
    def __call__ (self) -> Tuple:

        # Filter the ground true picks
        #self.filter_picks_DF()

        # Compute the distance of S picks
        all_dists_s = self.compare_PhaseNet_catalog_S_picks ()

        # Compute the distance of P picks
        all_dists_p = self.compare_PhaseNet_catalog_P_picks ()
        return all_dists_p, all_dists_s
    
    def filter_picks_DF (self):

        '''
        This function apply filter on existing picks DataFrame according to 
        start_year_analysis, end_year_analysis, start_day_analysis, end_day_analysis.

            Output:
                    - catalog_DF_P_picks (dataframe): catalog P picks dataframe
                    - catalog_DF_S_picks (dataframe): catalog S picks dataframe
            '''
        # load ground true picks
        DF_picks = self.Ground_truth

        # convert the Day of Year in Python to Month/Day
        start_date = datetime.datetime.strptime('{} {}'.format(self.start_day_analysis, self.start_year_analysis),'%j %Y')
        end_date   = datetime.datetime.strptime('{} {}'.format(self.end_day_analysis, self.end_year_analysis),'%j %Y')

        start_date_obspy = obspy.UTCDateTime(year=self.start_year_analysis, month=start_date.month, day=start_date.day, strict=False)
        end_date_obspy = obspy.UTCDateTime(year=self.end_year_analysis, month=end_date.month, day=end_date.day, hour=24, strict=False)

 
        catalog_DF_P_picks = DF_picks[(DF_picks['picks_time']>= start_date_obspy) & (DF_picks['picks_time']<=end_date_obspy) & (DF_picks['phase_hint']=='P')]
        catalog_DF_S_picks = DF_picks[(DF_picks['picks_time']>= start_date_obspy) & (DF_picks['picks_time']<=end_date_obspy) & (DF_picks['phase_hint']=='S')]

        catalog_DF_P_picks.to_pickle('Ground_truth_p_picks.pkl')
        catalog_DF_S_picks.to_pickle('Ground_truth_s_picks.pkl')
        
        #return catalog_DF_P_picks, catalog_DF_S_picks


    def compare_PhaseNet_catalog_P_picks (self) -> np.ndarray:

        '''
        This function compares the result of phase picker and existing ground true.
        '''
        # catalog_DF_P_picks, df_P_picks
        #with open(os.path.join('{0}/{1}'.format(seisbench.cache_root,'datasets/chile/parameters_tunning'), self.file_name),'rb') as fp:
        #    pick_df = pickle.load(fp)
        pick_df = self.event_picks

        if 'phase_hint' in pick_df.columns:
            pick_df.rename(columns={'phase_hint': 'type'}, inplace=True)
            pick_df['type'] = pick_df['type'].str.lower()
        
        df_P_picks = pick_df[pick_df.type == 'p']
        #with open('Ground_truth_p_picks','rb') as fs:
        #    catalog_DF_P_picks = pickle.load(fs)

        catalog_DF_P_picks = self.Ground_truth[self.Ground_truth.phase_hint=='P']
        # creat extra columns
        if 'id' in df_P_picks.columns:
            df_P_picks[['station_code', 'others']] = df_P_picks['id'].str.split('.', 1, expand=True)
            df_P_picks[['station_code', 'others']] = df_P_picks['others'].str.split('.', 1, expand=True)
            df_P_picks = df_P_picks.drop(['others'], axis=1)

        if 'timestamp' not in df_P_picks.columns:
            df_P_picks.rename(columns={'picks_time': 'timestamp'}, inplace=True)

        # find common station_code in catalog and PhaseNet
        boolean_column = catalog_DF_P_picks['station_code'].isin(df_P_picks['station_code'])
        catalog_DF_P_picks = catalog_DF_P_picks[(boolean_column==True)]
        all_dists = np.array([])
        common_stations = catalog_DF_P_picks['station_code'].unique()

        # Creat an empty DataFrame file to store all UTC time of PhaseNet in common station
        all_p_picks_exist_in_catalogtory = pd.DataFrame(index =[])

        # loop over all common station
        for i in common_stations:
            bo = catalog_DF_P_picks['station_code']==i
            catalog_filter_station = catalog_DF_P_picks[(bo==True)]
            ao = df_P_picks['station_code']==i
            phasenet_filter_station = df_P_picks[(ao==True)]

            # Convert UTC time to datetime64[ms] (millisecond)
            a = catalog_filter_station.picks_time.to_numpy(dtype='datetime64[ms]')[:, np.newaxis].astype("float")
            b = phasenet_filter_station.timestamp.to_numpy(dtype='datetime64[ms]')[:, np.newaxis].astype("float")

            #a[:, np.newaxis, :] - b[np.newaxis, :, :]
            
            # Calculate P1 norme of all datetime64[ms]
            #dist_mat = distance_matrix(a,b, p=1)
            #dists = np.min(dist_mat1, axis=1)

            dist_mat = np.sum((a[:,None]-b[:]),axis=-1)
            min_arg = np.argmin(abs(dist_mat), axis=1)
            dists = dist_mat[np.arange(dist_mat.shape[0]),min_arg]
            
            all_dists = np.append(all_dists, dists)

            # append phasenet_filter_station
            min_index = np.argmin(dist_mat, axis=1)

            #phasenet_filter_station = phasenet_filter_station[min_index]
            phasenet_filter_station = phasenet_filter_station.iloc[min_index,:]

            all_p_picks_exist_in_catalogtory = pd.concat([all_p_picks_exist_in_catalogtory, phasenet_filter_station], axis=0)

        return all_dists

        

    def compare_PhaseNet_catalog_S_picks (self) -> np.ndarray:

        '''
        This function compares the result of phase picker and existing ground true.
        '''
        # catalog_DF_P_picks, df_P_picks
        # catalog_DF_P_picks, df_P_picks
        #with open(os.path.join('{0}/{1}'.format(seisbench.cache_root,'datasets/chile/parameters_tunning'), self.file_name),'rb') as fp:
        #    pick_df = pickle.load(fp)
        pick_df = self.event_picks

        if 'phase_hint' in pick_df.columns:
            pick_df.rename(columns={'phase_hint': 'type'}, inplace=True)
            pick_df['type'] = pick_df['type'].str.lower()

        df_S_picks = pick_df[pick_df.type == 's']
        catalog_DF_S_picks = self.Ground_truth[self.Ground_truth.phase_hint=='S']

        # creat extra columns
        if 'id' in df_S_picks.columns:
            df_S_picks[['station_code', 'others']] = df_S_picks['id'].str.split('.', 1, expand=True)
            df_S_picks[['station_code', 'others']] = df_S_picks['others'].str.split('.', 1, expand=True)
            df_S_picks = df_S_picks.drop(['others'], axis=1)
        
        if 'timestamp' not in df_S_picks.columns:
            df_S_picks.rename(columns={'picks_time': 'timestamp'}, inplace=True)

        # find common station_code in catalog and PhaseNet
        boolean_column = catalog_DF_S_picks['station_code'].isin(df_S_picks['station_code'])
        catalog_DF_S_picks = catalog_DF_S_picks[(boolean_column==True)]
        all_dists = np.array([])
        common_stations = catalog_DF_S_picks['station_code'].unique()

        # Creat an empty DataFrame file to store all UTC time of PhaseNet in common station
        all_s_picks_exist_in_catalogtory = pd.DataFrame(index =[])
        
        # loop over all common statio
        for i in common_stations:
            bo = catalog_DF_S_picks['station_code']==i
            catalog_filter_station = catalog_DF_S_picks[(bo==True)]
            ao = df_S_picks['station_code']==i
            phasenet_filter_station = df_S_picks[(ao==True)]

            # Convert UTC time to datetime64[ms] (millisecond)
            a = catalog_filter_station.picks_time.to_numpy(dtype='datetime64[ms]')[:, np.newaxis].astype("float")
            b = phasenet_filter_station.timestamp.to_numpy(dtype='datetime64[ms]')[:, np.newaxis].astype("float")

            # Calculate P1 norme of all datetime64[m
            #dist_mat = distance_matrix(a,b, p=1)
            #dists = np.min(dist_mat, axis=1)

            dist_mat = np.sum((a[:,None]-b[:]),axis=-1)
            min_arg = np.argmin(abs(dist_mat), axis=1)
            dists = dist_mat[np.arange(dist_mat.shape[0]),min_arg]

            all_dists = np.append(all_dists, dists)

            # append phasenet_filter_station
            min_index = np.argmin(dist_mat, axis=1)

            #phasenet_filter_station = phasenet_filter_station[min_index]
            phasenet_filter_station = phasenet_filter_station.iloc[min_index,:]

            all_s_picks_exist_in_catalogtory = pd.concat([all_s_picks_exist_in_catalogtory, phasenet_filter_station], axis=0)


        return all_dists

if __name__ == "__main__":



    start_year_analysis = 2011
    start_day_analysis = 90
    end_year_analysis = 2011
    end_day_analysis = 90
    time_lag_threshold = 500 # mi second

    catalog = 'IPOC'
    GT = 'Hand-picked'
    title = '{0}{1}{2}{3}{4}'.format(catalog,' catalog', ' and Ground truth (',GT,') Comparison' )

    if GT =='IPOC':

        Ground_truth_file_path_p = '/home/javak/Sample_data_chile/Comparing PhaseNet and Catalog/EQTransformer_transfer learning_instance/Binary entropy/P=0.075, s= 0.1/2011.90 (trained on Iquque)/catalog_p_picks.pkl'
        with open(Ground_truth_file_path_p,'rb') as fp:
            Ground_truth_p = pickle.load(fp)

        Ground_truth_file_path_s = '/home/javak/Sample_data_chile/Comparing PhaseNet and Catalog/EQTransformer_transfer learning_instance/Binary entropy/P=0.075, s= 0.1/2011.90 (trained on Iquque)/catalog_s_picks.pkl'
        with open(Ground_truth_file_path_s,'rb') as fp:
            Ground_truth_s = pickle.load(fp)
        

        Ground_truth = pd.concat([Ground_truth_p, Ground_truth_s], axis=0)
        Ground_truth.sort_values(by=['picks_time'], inplace=True)

        Ground_truth.drop(columns=['picks_uncertainty','origins_time', 'origins_longitude', 'origins_latitude','magnitudes'], errors='ignore', inplace=True)

    if GT =='Hand-picked':
        Ground_truth_file_path = '/home/javak/Sample_data_chile/Events_catalog/Manual picks/Jonas/picks_2011_090_cleaned.pkl'
        with open(Ground_truth_file_path,'rb') as fp:
            Ground_truth = pickle.load(fp)



    if catalog =='IPOC':

        catalog_file_path_p = '/home/javak/Sample_data_chile/Comparing PhaseNet and Catalog/EQTransformer_transfer learning_instance/Binary entropy/P=0.075, s= 0.1/2011.90 (trained on Iquque)/catalog_p_picks.pkl'
        with open(catalog_file_path_p,'rb') as fp:
            catalog_p = pickle.load(fp)

        catalog_file_path_s = '/home/javak/Sample_data_chile/Comparing PhaseNet and Catalog/EQTransformer_transfer learning_instance/Binary entropy/P=0.075, s= 0.1/2011.90 (trained on Iquque)/catalog_s_picks.pkl'
        with open(catalog_file_path_s,'rb') as fp:
            catalog_s = pickle.load(fp)
    

    catalog = pd.concat([catalog_p, catalog_s], axis=0)
    catalog.sort_values(by=['picks_time'], inplace=True)

    catalog.drop(columns=['picks_uncertainty','origins_time', 'origins_longitude', 'origins_latitude','magnitudes'], errors='ignore', inplace=True)



    if catalog == 'Hand-picked':

        catalog_file_path = '/home/javak/Sample_data_chile/Events_catalog/Manual picks/Jonas/picks_2011_090_cleaned.pkl'
        with open(catalog_file_path,'rb') as fp:
            event_picks = pickle.load(fp)


    if catalog == 'Instance-Iquique':

        picker_p_picks_file_path = '/home/javak/Sample_data_chile/Comparing PhaseNet and Catalog/EQTransformer_transfer learning_instance/Binary entropy/P=0.075, s= 0.1/2011.90 (trained on Iquque)/PhaseNet_result_p_picks.pkl'
        with open(picker_p_picks_file_path,'rb') as fp:
            p_picks = pickle.load(fp)

        picker_s_picks_file_path = '/home/javak/Sample_data_chile/Comparing PhaseNet and Catalog/EQTransformer_transfer learning_instance/Binary entropy/P=0.075, s= 0.1/2011.90 (trained on Iquque)/PhaseNet_result_s_picks.pkl'
        with open(picker_s_picks_file_path,'rb') as fp:
            s_picks = pickle.load(fp)

        event_picks = pd.concat([p_picks,s_picks], axis=0)
        event_picks.sort_values(by=['timestamp'], inplace=True)


    if catalog == 'Instance':

        picker_p_picks_file_path = '/home/javak/Sample_data_chile/Comparing PhaseNet and Catalog/EQ Trasfermer based on instance/2011.90/PhaseNet_result_p_picks.pkl'
        with open(picker_p_picks_file_path,'rb') as fp:
            p_picks = pickle.load(fp)

        picker_s_picks_file_path = '/home/javak/Sample_data_chile/Comparing PhaseNet and Catalog/EQ Trasfermer based on instance/2011.90/PhaseNet_result_s_picks.pkl'
        with open(picker_s_picks_file_path,'rb') as fp:
            s_picks = pickle.load(fp)

        event_picks = pd.concat([p_picks,s_picks], axis=0)
        event_picks.sort_values(by=['timestamp'], inplace=True)

    '''
    # Loading Ground truth data
    picker_p_picks_file_path = '/home/javak/Sample_data_chile/Comparing PhaseNet and Catalog/EQTransformer_transfer learning_instance/Binary entropy/P=0.075, s= 0.1/2011.90 (trained on Iquque)/catalog_p_picks.pkl'
    with open(picker_p_picks_file_path,'rb') as fp:
        picker_p_picks_file = pickle.load(fp)

    picker_s_picks_file_path = '/home/javak/Sample_data_chile/Comparing PhaseNet and Catalog/EQTransformer_transfer learning_instance/Binary entropy/P=0.075, s= 0.1/2011.90 (trained on Iquque)/catalog_s_picks.pkl'
    with open(picker_s_picks_file_path,'rb') as fp:
        picker_s_picks_file = pickle.load(fp)



    event_picks = pd.concat([picker_p_picks_file, picker_s_picks_file], axis=0)
    event_picks.sort_values(by=['picks_time'], inplace=True)

    event_picks.drop(columns=['picks_uncertainty','origins_time', 'origins_longitude', 'origins_latitude','magnitudes'], errors='ignore', inplace=True)
    '''
    # Loading automatics picker data

    # Loading automatics picker data
    #picker_p_picks_file_path = '/home/javak/Sample_data_chile/Comparing PhaseNet and Catalog/EQTransformer_transfer learning_instance/Binary entropy/P=0.075, s= 0.1/2011.90 (trained on Iquque)/catalog_p_picks.pkl'
    #with open(picker_p_picks_file_path,'rb') as fp:
    #    p_picks = pickle.load(fp)

    #picker_s_picks_file_path = '/home/javak/Sample_data_chile/Comparing PhaseNet and Catalog/EQTransformer_transfer learning_instance/Binary entropy/P=0.075, s= 0.1/2011.90 (trained on Iquque)/catalog_p_picks.pkl'
    #with open(picker_s_picks_file_path,'rb') as fp:
    #    s_picks = pickle.load(fp)

    #event_picks = pd.concat([p_picks,s_picks], axis=0)
    #event_picks.sort_values(by=['timestamp'], inplace=True)
    #event_picks_file_path = '/home/javak/Sample_data_chile/Events_catalog/Manual picks/Jonas/picks_2011_090_cleaned.pkl'
    #with open(event_picks_file_path,'rb') as fp:
    #    event_picks = pickle.load(fp)

    picks_obj = Picks_Comparison (start_year_analysis, 
                    start_day_analysis,
                    end_year_analysis,
                    end_day_analysis,event_picks, Ground_truth)

    all_dists_p, all_dists_s = picks_obj()
    '''
    start_year_analysis = 2011
    start_day_analysis = 90
    end_year_analysis = 2011
    end_day_analysis = 90
    time_lag_threshold = 500


    Ground_truth_file_path ='/home/javak/Sample_data_chile/Events_catalog/Manual picks/Jonas/picks_2011_090_cleaned.pkl'
    with open(Ground_truth_file_path,'rb') as fp:
        Ground_truth = pickle.load(fp)
    

    # Loading automatics picker data
    picker_p_picks_file_path = '/home/javak/Sample_data_chile/Comparing PhaseNet and Catalog/EQTransformer_transfer learning_instance/Binary entropy/P=0.075, s= 0.1/2011.90 (trained on Iquque)/catalog_p_picks.pkl'
    with open(picker_p_picks_file_path,'rb') as fp:
        p_picks = pickle.load(fp)

    picker_s_picks_file_path = '/home/javak/Sample_data_chile/Comparing PhaseNet and Catalog/EQTransformer_transfer learning_instance/Binary entropy/P=0.075, s= 0.1/2011.90 (trained on Iquque)/catalog_p_picks.pkl'
    with open(picker_s_picks_file_path,'rb') as fp:
        s_picks = pickle.load(fp)

    event_picks = pd.concat([p_picks,s_picks], axis=0)
    event_picks.sort_values(by=['picks_time'], inplace=True)
    
    # Loading automatics picker data
    catalog_p_picks_file_path = '/home/javak/Sample_data_chile/Comparing PhaseNet and Catalog/EQTransformer_transfer learning_instance/Binary entropy/P=0.075, s= 0.1/2011.90 (trained on Iquque)/catalog_p_picks.pkl'
    with open(catalog_p_picks_file_path,'rb') as fp:
        catalog_p_picks = pickle.load(fp)

    catalog_s_picks_file_path = '/home/javak/Sample_data_chile/Comparing PhaseNet and Catalog/EQTransformer_transfer learning_instance/Binary entropy/P=0.075, s= 0.1/2011.90 (trained on Iquque)/catalog_s_picks.pkl'
    with open(catalog_s_picks_file_path,'rb') as fp:
        catalog_s_picks = pickle.load(fp)



    # Loading Ground truth data
    Ground_truth_file_path_p = '/home/javak/Sample_data_chile/Comparing PhaseNet and Catalog/EQTransformer_transfer learning_instance/Binary entropy/P=0.075, s= 0.1/2011.90 (trained on Iquque)/catalog_p_picks.pkl'
    with open(Ground_truth_file_path_p,'rb') as fp:
        Ground_truth_p = pickle.load(fp)

    Ground_truth_file_path_s = '/home/javak/Sample_data_chile/Comparing PhaseNet and Catalog/EQTransformer_transfer learning_instance/Binary entropy/P=0.075, s= 0.1/2011.90 (trained on Iquque)/catalog_s_picks.pkl'
    with open(Ground_truth_file_path_s,'rb') as fp:
        Ground_truth_s = pickle.load(fp)
    

    Ground_truth = pd.concat([Ground_truth_p, Ground_truth_s], axis=0)
    Ground_truth.sort_values(by=['picks_time'], inplace=True)

    Ground_truth.drop(columns=['picks_uncertainty','origins_time', 'origins_longitude', 'origins_latitude','magnitudes'], errors='ignore', inplace=True)
    # Loading automatics picker data
    
    picker_p_picks_file_path = '/home/javak/Sample_data_chile/Comparing PhaseNet and Catalog/EQ Trasfermer based on instance/2011.90/PhaseNet_result_p_picks.pkl'
    with open(picker_p_picks_file_path,'rb') as fp:
        p_picks = pickle.load(fp)

    picker_s_picks_file_path = '/home/javak/Sample_data_chile/Comparing PhaseNet and Catalog/EQ Trasfermer based on instance/2011.90/PhaseNet_result_s_picks.pkl'
    with open(picker_s_picks_file_path,'rb') as fp:
        s_picks = pickle.load(fp)

    event_picks = pd.concat([p_picks,s_picks], axis=0)
    event_picks.sort_values(by=['timestamp'], inplace=True)
    
    event_picks_file_path = '/home/javak/Sample_data_chile/Events_catalog/Manual picks/Jonas/picks_2011_090_cleaned.pkl'
    with open(event_picks_file_path,'rb') as fp:
        event_picks = pickle.load(fp)

    picks_obj = Picks_Comparison (start_year_analysis, 
                    start_day_analysis,
                    end_year_analysis,
                    end_day_analysis,event_picks, Ground_truth)

    all_dists_p, all_dists_s = picks_obj()

    print(all_dists_p.shape)
    print(all_dists_s.shape)
    print(event_picks.shape)
    print(Ground_truth.shape)

    print(all_dists_p[np.abs(all_dists_p) < time_lag_threshold].shape[0]/all_dists_p.shape[0])
    print(all_dists_s[np.abs(all_dists_s) < time_lag_threshold].shape[0]/all_dists_s.shape[0])
    '''



    

