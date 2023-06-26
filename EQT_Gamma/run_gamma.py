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
        pick_df.sort_values("timestamp")

    #pick_df = pick_df.iloc[0:500]
    # Run Gamma associator
    catalogs, assignments = association(pick_df, station_df, config, method=config["method"])

    # sort catalogs
    ind = np.argsort([catalogs[idx]['time'] for idx in range(len(catalogs))])
    catalogs = [catalogs[idx] for idx in ind]

    #for i in range(len(catalogs)):
    #    timestamp_integer = int(datetime.strptime(catalogs[i]['time'], "%Y-%m-%dT%H:%M:%S.%f").timestamp())
    
    
    #catalogs.sort(key=lambda x: catalogs['time'])

    catalog_df = pd.DataFrame(catalogs)
    assignments = pd.DataFrame(assignments, columns=["pick_idx", "event_idx", "prob_gamma"])
    
    # Convert to Snuffler format
    Snuffler_obj = SnufflerConvertor(opt)
    Snuffler_obj(pick_df,catalog_df, catalogs, assignments)

    vis = Visualization ()

    # Create a stream object
    stream = Data_Preprocessing(opt.time_interval[0],opt.time_interval[1],opt.time_interval[2],opt.time_interval[3]).get_waveforms_chile()
    
    # plot catalog
    vis.plot_catalog(catalog_df, station_df )

    # plot waveform with associated events
    #vis.plot_waveform(picks, catalog, assignments, stream, station_dict)

    vis.plot_sorted_lat(picks, catalog_df, assignments, stream, station_dict)


    # In case of occuring an thread association, it it is better to use ulimit -u 8192
    # to increase the number of limitation.


    








