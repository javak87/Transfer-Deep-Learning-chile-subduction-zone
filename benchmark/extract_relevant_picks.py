import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class extract_relevant_picks (object):

    def __init__ (self, files_address:'str'):

        '''
        This class goes through pickle files and select the picks files among different files.
        Then, return those picks which are associated.

            - files address: the directory of tunning files 
        '''

        self.files_address = files_address

        tuning_files = os.listdir(self.files_address)

        # select the catalog_files
        self.catalog_files = list(filter(lambda k: 'catalog_' in k, tuning_files))
        self.catalog_files_compare = [elem.replace('catalog_', '') for elem in self.catalog_files]
        
        # select the assignment_files
        self.assignment_files = list(filter(lambda k: 'assignments_' in k, tuning_files))
        self.assignment_files_compare  = [elem.replace('assignments_', '') for elem in self.assignment_files]

    def extract_picks (self,picks_file_name:'str') -> pd.DataFrame:

        # select files contains "picks"
        picks_files_compare = picks_file_name.replace('picks_', '')

        # select the relevant catalog files realted the the picks file
        catalog_file = self.catalog_files[self.catalog_files_compare.index(picks_files_compare)]

        # select the relevant assignment files realted the the picks file
        assignments_file = self.assignment_files [self.assignment_files_compare.index(picks_files_compare)]

        # read the relevant catalog file
        with open(os.path.join(self.files_address, catalog_file),'rb') as fp:
            catalog = pickle.load(fp)
        
        # read the relevant catalog file
        with open(os.path.join(self.files_address, assignments_file),'rb') as fe:
            assignments = pickle.load(fe)
        
        # read the relevant picks file
        with open(os.path.join(self.files_address, picks_file_name),'rb') as fs:
            picks = pickle.load(fs)
        
        # create a data frame
        selected_picks = pd.DataFrame(columns = ['id', 'timestamp', 'prob', 'type'])

        # Extract those picks which are existed in Catalog
        for event_idx in catalog.event_idx.index:

            event_picks = picks.iloc[assignments[assignments["event_idx"] == event_idx]["pick_idx"]]
            selected_picks = pd.concat([selected_picks, event_picks])
        
        return selected_picks

if __name__ == "__main__":

    tuning_files_add = '/home/javak/.seisbench/datasets/chile/parameters_tunning'
    picks_file_name = 'picks_min_pk_eq:4_meth:BGMM_amp:False_Pth:0.5_Sth:0.5'
    obj = extract_relevant_picks(tuning_files_add)
    c = obj.extract_picks(picks_file_name)


