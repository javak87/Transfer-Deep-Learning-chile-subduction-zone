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

from gamma.utils import association
import seisbench.models as sbm
from options.gamma_options import GammaOptions
from utils.config_gamma import ConfigGamma
from utils.data_preprocessing import Data_Preprocessing
from utils.visualization import Visualization
from utils.picks_df_creator import picks_df_creator



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

    pick_df = pick_df.iloc[0:1000]
    # Run Gamma associator
    catalogs, assignments = association(pick_df, station_df, config, method=config["method"])
    catalog = pd.DataFrame(catalogs)
    assignments = pd.DataFrame(assignments, columns=["pick_idx", "event_idx", "prob_gamma"])
    
    vis = Visualization ()

    # Create a stream object
    stream = Data_Preprocessing(opt.time_interval[0],opt.time_interval[1],opt.time_interval[2],opt.time_interval[3]).get_waveforms_chile()
    
    # plot catalog
    vis.plot_catalog(catalog, station_df )

    # plot waveform with associated events
    vis.plot_waveform(picks, catalog, assignments, stream, station_dict)

    








