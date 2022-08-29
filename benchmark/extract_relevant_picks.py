import obspy
import os
import pickle
import datetime
import seisbench
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from picks_comparison import Picks_Comparison
import warnings
warnings.filterwarnings('ignore')



class extract_relevant_picks (object):

    def __init__ (self, files_address):

        self.files_address = files_address

        tuning_files = os.listdir(self.files_address)

        self.catalog_files = list(filter(lambda k: 'catalog_' in k, tuning_files))
        self.catalog_files_compare = [elem.replace('catalog_', '') for elem in self.catalog_files]

        self.assignment_files = list(filter(lambda k: 'assignments_' in k, tuning_files))
        self.assignment_files_compare  = [elem.replace('assignments_', '') for elem in self.assignment_files]

    def extract_picks (self,picks_file_name):

        picks_files_compare = picks_file_name.replace('picks_', '')

        catalog_file = self.catalog_files[self.catalog_files_compare.index(picks_files_compare)]

        assignments_file = self.assignment_files [self.assignment_files_compare.index(picks_files_compare)]

        with open(os.path.join(self.files_address, catalog_file),'rb') as fp:
            catalog = pickle.load(fp)
        
        with open(os.path.join(self.files_address, assignments_file),'rb') as fe:
            assignments = pickle.load(fe)
        
        with open(os.path.join(self.files_address, picks_file_name),'rb') as fs:
            picks = pickle.load(fs)
        
        selected_picks = pd.DataFrame(columns = ['id', 'timestamp', 'prob', 'type'])

        for event_idx in catalog.event_idx.index:

            event_picks = picks.iloc[assignments[assignments["event_idx"] == event_idx]["pick_idx"]]
            selected_picks = pd.concat([selected_picks, event_picks])
        
        return selected_picks




    
if __name__ == "__main__":

    tuning_files_add = '/home/javak/.seisbench/datasets/chile/parameters_tunning'
    picks_file_name = 'picks_min_pk_eq:4_meth:BGMM_amp:False_Pth:0.5_Sth:0.5'
    obj = extract_relevant_picks(tuning_files_add)
    c = obj.extract_picks(picks_file_name)


