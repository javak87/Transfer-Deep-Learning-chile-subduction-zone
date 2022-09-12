from re import X
from typing import Tuple
import pandas as pd
import numpy as np
import os
import pickle
from scipy.spatial import distance_matrix
import seisbench

class Evaluation (object):

    def __init__(self, event_picks: 'pd.DataFrame')-> None:
        
        '''
        This class is used to evalaute the picks quality of the picker based on the ground true.
        Parameters:
                - event_picks (data frame): data frame of phasenet picks
                - time_lag_threshold: time lag threshold
        '''
        self.event_picks = event_picks
        self.time_lag_threshold = 500

    def proximity_matrix (self, phase_hint:'str') -> Tuple:
        '''
        This function measure the proximity matrix of PhaseNet performance assuming that the catalog is the ground-true.
        In order to measure the proximity matrix, the following variables are using:

                - True Positive (Tp): 
                                    1- Peak probabilities of the phase is above 0.5 
                                    2- Arrival-time residuals is less than 0.1 second

                - False Positive (Fp):
                                    1- Peak probabilities of the phase is above 0.5 
                                    2- Arrival-time residuals is more than 0.1 second

                - True Negative (Tn): 
                                    1- Peak probabilities of the phase is below 0.5 
                                    2- Arrival-time residuals is less than 0.1 second

                - False Negative (Fn):
                                    1- Peak probabilities of the phase is below 0.5 
                                    2- Arrival-time residuals is more than 0.1 second            
        

                parameters:
                        - Phase_hint (S or P): phase hint

                Outputs:
                        - Precision
                        - recall
                        - F1 score
        '''
        if phase_hint == 'P':

            phasenet_DF = self.event_picks[self.event_picks.type=='p']
            phasenet_DF['picks_time']= phasenet_DF.timestamp

            # load catalog_p_picks.pkl
            with open(os.path.join('{0}/{1}'.format(seisbench.cache_root,'datasets/chile/files_paths'), 'catalog_p_picks.pkl'),'rb') as fs:
                catalog_DF = pickle.load(fs)
            
        else:
            # load PhaseNet_result_s_picks.pkl
            phasenet_DF = self.event_picks[self.event_picks.type=='s']
            # add picks_time to PhaseNet_result_s_picks data frame for synchronization
            phasenet_DF['picks_time']= phasenet_DF.timestamp

            # load catalog_s_picks.pkl
            with open(os.path.join('{0}/{1}'.format(seisbench.cache_root,'datasets/chile/files_paths'), 'catalog_s_picks.pkl'),'rb') as fs:
                catalog_DF = pickle.load(fs)
        # creat extra columns
        phasenet_DF[['network', 'others']] = phasenet_DF['id'].str.split('.', 1, expand=True)
        phasenet_DF[['station_code', 'date']] = phasenet_DF['others'].str.split('.', 1, expand=True)
        phasenet_DF = phasenet_DF.drop(['date', 'others'], axis=1)

        # Intialize the true postive counter, false postive counter, true negative counter, and false negative counter 
        true_pos_count = 0
        false_pos_count = 0

        true_neg_count = 0
        false_neg_count = 0
        all = 0

        # Find common station code in phasenet_DF and catalog_DF
        common_station = np.intersect1d(catalog_DF.station_code.unique(), phasenet_DF.station_code.unique())
        
        # loop over all common stations
        for i in common_station:
            bo = catalog_DF['station_code']==i
            catalog_filter_station = catalog_DF[(bo==True)]
            ao = phasenet_DF['station_code']==i
            phasenet_filter_station = phasenet_DF[(ao==True)]

            # Convert UTC time to datetime64[ms] (millisecond)
            a = catalog_filter_station.picks_time.to_numpy(dtype='datetime64[ms]')[:, np.newaxis].astype("float")
            b = phasenet_filter_station.timestamp.to_numpy(dtype='datetime64[ms]')[:, np.newaxis].astype("float")

            # Calculate P1 norme of all datetime64[m
            dist_mat = distance_matrix(a,b, p=1)

            ##
            dists = np.min(dist_mat, axis=1)

            all = all + catalog_filter_station.shape[0]
            probability = phasenet_filter_station.prob.iloc[np.argmin(dist_mat, axis=1)].to_numpy()

            #calculate True poitive
            Tp = catalog_filter_station[(dists <= self.time_lag_threshold) & (probability >= 0.5)].shape[0]
            true_pos_count = true_pos_count + Tp

            # calculate false poitive
            Fp = catalog_filter_station[(dists > self.time_lag_threshold) & (probability >= 0.5)].shape[0]
            false_pos_count = false_pos_count + Fp

            # calculate true negative
            Tn = catalog_filter_station[(dists > self.time_lag_threshold) & (probability < 0.5)].shape[0]
            true_neg_count = true_neg_count + Tn

            # calculate false negative

            Fn = catalog_filter_station[(dists <= self.time_lag_threshold) & (probability < 0.5)].shape[0]
            false_neg_count = false_neg_count + Fn

        # calculate Precision
        precision = true_pos_count /(true_pos_count + false_pos_count)

        precision= round(precision,4)
        # calculate Recall
        recall = true_pos_count /(true_pos_count + false_neg_count)
        recall= round(recall,4)
        # calculate f1 score
        f1_score = (2*precision*recall)/(precision + recall)
        f1_score= round(f1_score,4)


        return precision, recall, f1_score