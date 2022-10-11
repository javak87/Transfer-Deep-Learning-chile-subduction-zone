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
                    end_year_analysis, end_day_analysis, event_picks):

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
        
    def __call__ (self) -> Tuple:

        # Filter the ground true picks
        self.filter_picks_DF()

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
        with open(os.path.join('{0}/{1}'.format(seisbench.cache_root,'datasets/chile/files_paths'), 'day1.pkl'),'rb') as fp:
            DF_picks = pickle.load(fp)

        DF_picks = DF_picks[(DF_picks.origins_latitude < -19) & (DF_picks.origins_latitude > -21.5) & (DF_picks.origins_longitude < -69)]
        # convert the Day of Year in Python to Month/Day
        start_date = datetime.datetime.strptime('{} {}'.format(self.start_day_analysis, self.start_year_analysis),'%j %Y')
        end_date   = datetime.datetime.strptime('{} {}'.format(self.end_day_analysis, self.end_year_analysis),'%j %Y')

        start_date_obspy = obspy.UTCDateTime(year=self.start_year_analysis, month=start_date.month, day=start_date.day, strict=False)
        end_date_obspy = obspy.UTCDateTime(year=self.end_year_analysis, month=end_date.month, day=end_date.day, hour=24, strict=False)

 
        catalog_DF_P_picks = DF_picks[(DF_picks['picks_time']>= start_date_obspy) & (DF_picks['picks_time']<=end_date_obspy) & (DF_picks['phase_hint']=='P')]
        catalog_DF_S_picks = DF_picks[(DF_picks['picks_time']>= start_date_obspy) & (DF_picks['picks_time']<=end_date_obspy) & (DF_picks['phase_hint']=='S')]

        catalog_DF_P_picks.to_pickle(os.path.join('{0}/{1}'.format(seisbench.cache_root,'datasets/chile/files_paths'), 'catalog_p_picks.pkl'))
        catalog_DF_S_picks.to_pickle(os.path.join('{0}/{1}'.format(seisbench.cache_root,'datasets/chile/files_paths'), 'catalog_s_picks.pkl'))
        
        #return catalog_DF_P_picks, catalog_DF_S_picks


    def compare_PhaseNet_catalog_P_picks (self) -> np.ndarray:

        '''
        This function compares the result of phase picker and existing ground true.
        '''
        # catalog_DF_P_picks, df_P_picks
        #with open(os.path.join('{0}/{1}'.format(seisbench.cache_root,'datasets/chile/parameters_tunning'), self.file_name),'rb') as fp:
        #    pick_df = pickle.load(fp)
        pick_df = self.event_picks
        df_P_picks = pick_df[pick_df.type == 'p']
        with open(os.path.join('{0}/{1}'.format(seisbench.cache_root,'datasets/chile/files_paths'), 'catalog_p_picks.pkl'),'rb') as fs:
            catalog_DF_P_picks = pickle.load(fs)

        # creat extra columns
        df_P_picks[['station_code', 'others']] = df_P_picks['id'].str.split('.', 1, expand=True)
        df_P_picks[['station_code', 'others']] = df_P_picks['others'].str.split('.', 1, expand=True)
        df_P_picks = df_P_picks.drop(['others'], axis=1)

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

        df_S_picks = pick_df[pick_df.type == 's']
        with open(os.path.join('{0}/{1}'.format(seisbench.cache_root,'datasets/chile/files_paths'), 'catalog_s_picks.pkl'),'rb') as fs:
            catalog_DF_S_picks = pickle.load(fs)

        # creat extra columns
        df_S_picks[['station_code', 'others']] = df_S_picks['id'].str.split('.', 1, expand=True)
        df_S_picks[['station_code', 'others']] = df_S_picks['others'].str.split('.', 1, expand=True)
        df_S_picks = df_S_picks.drop(['others'], axis=1)

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


    start_year_analysis = 2012
    start_day_analysis = 1
    end_year_analysis = 2012
    end_day_analysis = 1

    obj = Picks_Comparison (start_year_analysis, 
                    start_day_analysis,
                    end_year_analysis,
                    end_day_analysis)
    
    dists = obj()
    b= 1