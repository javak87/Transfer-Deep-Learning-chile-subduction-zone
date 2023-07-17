import obspy
import pickle
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from pyproj import CRS, Transformer
from datetime import datetime

from gamma.utils import association
import seisbench.models as sbm
from options.gamma_options import GammaOptions
from utils.config_gamma import ConfigGamma
from utils.data_preprocessing import Data_Preprocessing
from utils.visualization import Visualization
from utils.picks_df_creator import picks_df_creator
from utils.snuffler_convertor import SnufflerConvertor
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'


if __name__ == '__main__':

    # -----  Loading the init options -----
    opt = GammaOptions().parse()

    config  = ConfigGamma(opt)()

    # return stations prop
    station_dict, station_df = Data_Preprocessing(opt.time_interval[0],opt.time_interval[1],opt.time_interval[2],opt.time_interval[3]).get_station(opt.client, opt.transformer)
    
    # open generated picks by EQT
    with open(opt.pick_path, 'rb') as fp:
        picks = pickle.load(fp)
        
        # create a data frame of picks
        pick_df = picks_df_creator(picks)
        pick_df.sort_values("timestamp", inplace=True)

    #pick_df = pick_df.iloc[0:500]
    # Run Gamma associator
    catalogs, assignments = association(pick_df, station_df, config, method=config["method"])

    # sort catalogs
    ind = np.argsort([catalogs[idx]['time'] for idx in range(len(catalogs))])
    catalogs = [catalogs[idx] for idx in ind]
    #assignments = [assignments[idx] for idx in ind]

    #for i in range(len(catalogs)):
    #    timestamp_integer = int(datetime.strptime(catalogs[i]['time'], "%Y-%m-%dT%H:%M:%S.%f").timestamp())
    
    
    #catalogs.sort(key=lambda x: catalogs['time'])

    catalog_df = pd.DataFrame(catalogs)
    assignments = pd.DataFrame(assignments, columns=["pick_idx", "event_idx", "prob_gamma"])

    # Sort assignments based on the time
    dt_list = [assignments[assignments['event_idx']==event_idx] for event_idx in catalog_df['event_index']]
    
    assignments = pd.concat(dt_list, keys=assignments['event_idx'])

    # Reset the index of the combined dataframe
    assignments = assignments.reset_index(level=0, drop=True).reset_index()

    #assignments = assignments.sort_values("event_ot")
    #assignments = assignments.drop(['event_ot'], axis=1)
    #assignments['event_idx'] = pd.Categorical(assignments['event_idx'], categories=ind, ordered=True)
    #assignments = assignments.sort_values('event_idx')    

    #[picks[i] for i in assignments[assignments["event_idx"] == event_idx]["pick_idx"]]
    #assignments.sort_values("pick_idx", inplace=True)
    # Convert to Snuffler format
    Snuffler_obj = SnufflerConvertor(opt)
    Snuffler_obj(pick_df,catalog_df, catalogs, assignments)

    b = 1
    '''
    vis = Visualization ()

    # Create a stream object
    stream = Data_Preprocessing(opt.time_interval[0],opt.time_interval[1],opt.time_interval[2],opt.time_interval[3]).get_waveforms_chile()
    
    # plot catalog
    vis.plot_catalog(catalog_df, station_df )

    # plot waveform with associated events
    #vis.plot_waveform(picks, catalog, assignments, stream, station_dict)

    vis.plot_sorted_lat(picks,catalogs, catalog_df, assignments, stream, station_dict)


    # In case of occuring an thread association, it it is better to use ulimit -u 8192
    # to increase the number of limitation.
    b = 1
    '''

    








