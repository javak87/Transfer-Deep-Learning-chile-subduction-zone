import os
from typing import Tuple
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from pyproj import CRS, Transformer
import pandas as pd
from tqdm import tqdm
import torch
import seisbench
from gamma.utils import association
import seisbench.models as sbm
from data_preprocessing import Data_Preprocessing
from datetime import datetime


class Parameter_Tuning (object):

    def __init__ (self, P_threshold_ls:'float', 
                        S_threshold_ls:'float', 
                        dbscan_min_samples_ls:'int',
                        min_picks_per_eq:'int', 
                        method:'str', 
                        use_amp:'bool',
                        start_year_analysis:'int',
                        start_day_analysis:'int',
                        end_year_analysis:'int',
                        end_day_analysis:'int'):

        self.P_threshold_ls = P_threshold_ls
        self.S_threshold_ls = S_threshold_ls
        self.dbscan_min_samples_ls = dbscan_min_samples_ls
        self.min_picks_per_eq = min_picks_per_eq
        self.method = method
        self.use_amp = use_amp

        self.start_year_analysis = start_year_analysis
        self.start_day_analysis = start_day_analysis
        self.end_year_analysis = end_year_analysis
        self.end_day_analysis = end_day_analysis
    
    def __call__ (self):

        '''
        This class is used to tune the Phasenet & Gamma parameters based on the grid search method.
        - Inputs:
            - P_threshold_ls (): P-picks threshold between 0 to 1.
            - S_threshold_ls: S-picks threshold between 0 to 1.
            - dbscan_min_samples_ls: The number of samples in a neighborhood for a point to be considered as a core point.
            - min_picks_per_eq: min_picks_per_eq: Minimum picks for associated earthquakes. 
                We can also specify minimum P or S picks:
                    min_p_picks_per_eq: Minimum P-picks for associated earthquakes.
                    min_s_picks_per_eq: Minimum S-picks for associated earthquakes.
            - method: "BGMM" or "GMM"
            - use_amp: Allow algorithm to use amplitue or not.

            to see more detail, please read the gamma documentation:
                https://github.com/wayneweiqiang/GaMMA
            
            - start_year_analysis = starting year of analysis
            - start_day_analysis = starting day of analysis
            - end_year_analysis = eding year of analysis
            - end_day_analysis = eding day of analysis
        
        Outputs: There are several output pikle files will be created in the 
            "/home/javak/.seisbench/datasets/chile/files_paths" directory. The name of files show the relevant tuned parameters.
        '''

        for P_th in self.P_threshold_ls:

            for S_th in self.S_threshold_ls:

                for min_samples in self.dbscan_min_samples_ls:

                    for min_pk_eq in min_picks_per_eq:

                        for meth in method:

                            for amp in use_amp:

                                # config gamma parameters based on the inputs
                                config  = self.config_association(min_samples, min_pk_eq, meth, amp)

                                # get stream based on the start_year_analysis, start_day_analysis, end_year_analysis, and end_day_analysis
                                stream, inv = self.get_waveforms()

                                # Run the phasenet based on the P_th, S_th, and stream
                                station_df, pick_df = self.phase_picking (stream, inv, P_th, S_th)
                                
                                # Run association
                                self.association (pick_df, station_df, config, min_samples, P_th, S_th, min_pk_eq, meth, amp)

                                print ('---------------------------------------------------')

                                now = datetime.now()
                                current_time = now.strftime("%H:%M:%S")

                                print("Current Time =", current_time)
                                print ('model with the following variables has been stored:')
                                print ('P_threshold: ', P_th)
                                print ('S_threshold: ', S_th)
                                print ('min_samples: ', min_samples)
                                print ('min_pk_eq: ', min_pk_eq)
                                print ('meth: ', meth)
                                print ('amp: ', amp)


    def config_association (self, min_samples, min_pk_eq, meth, amp) -> dict:
        
        '''
        This function has been used for configure the gamma association.
        '''
        # Gamma
        config = {}
        config["dims"] = ['x(km)', 'y(km)', 'z(km)']
        config["use_dbscan"] = True
        config["use_amplitude"] = amp
        config["x(km)"] = (250, 600)
        config["y(km)"] = (7200, 8000)
        config["z(km)"] = (0, 150)
        config["vel"] = {"p": 7.0, "s": 7.0 / 1.75}  # We assume rather high velocities as we expect deeper events
        config["method"] = meth
        if config["method"] == "BGMM":
            config["oversample_factor"] = 4
        if config["method"] == "GMM":
            config["oversample_factor"] = 1

        # DBSCAN
        config["bfgs_bounds"] = (
            (config["x(km)"][0] - 1, config["x(km)"][1] + 1),  # x
            (config["y(km)"][0] - 1, config["y(km)"][1] + 1),  # y
            (0, config["z(km)"][1] + 1),  # x
            (None, None),  # t
        )
        config["dbscan_eps"] = 30  # seconds
        config["dbscan_min_samples"] = min_samples

        # Filtering
        config["min_picks_per_eq"] = min_pk_eq
        config["max_sigma11"] = 2.0
        config["max_sigma22"] = 1.0
        config["max_sigma12"] = 1.0

        return config

    def get_waveforms (self) -> Tuple:

        '''
        This function extract relevant streams based on the start_year_analysis, start_day_analysis,
                end_year_analysis, end_day_analysis
        
        To get stream, some pikle files should be existed in the following directory:
            "/home/javak/.seisbench/datasets/chile/files_paths"
        
        Outputs:
            - stream: filtered streams
            - inv: station information
        '''
        obj = Data_Preprocessing (self.start_year_analysis, self.start_day_analysis,
                            self.end_year_analysis, self.end_day_analysis)

        stream = obj.get_waveforms_chile()
        client = Client("GFZ")
        
        # The start time and end time in this function just for getting station information.
        inv = client.get_stations(network="CX", station="*", location="*", channel="HH?", starttime=UTCDateTime("2014/05/01 00:00:00"), endtime=UTCDateTime("2014/05/01 00:00:00")+12*60)
        return stream, inv
    
    def phase_picking (self, stream, inv, P_th, S_th)-> Tuple:
        
        '''
        This function run the phasenet based on the given thresholds.
        Parameters:
                - stream: stream data feeding into the phasenet
                - inv: stations information
                - P_th: P picks thresholds
                - S_th: S picks thresholds
        
        Outputs:
                - pick_df (data frame): data frame of picks 
                - station_df (data frame): data frame of station information
        '''

        wgs84 = CRS.from_epsg(4326)
        local_crs = CRS.from_epsg(9155)  # SIRGAS-Chile 2016 / UTM zone 19S
        transformer = Transformer.from_crs(wgs84, local_crs)

        # Run the pretained phasenet version on "instance" dataset
        picker = sbm.PhaseNet.from_pretrained("instance")

        if torch.cuda.is_available():
            picker.cuda()

        # We tuned the thresholds a bit - Feel free to play around with these values
        #picks, _ = picker.classify(stream, batch_size=256, P_threshold=0.7, S_threshold=0.7, parallelism=1)
        
        picks = picker.classify(stream, batch_size=256, P_threshold=P_th, S_threshold=S_th, parallelism=1)
        
        pick_df = []
        for p in picks:
            pick_df.append({
                "id": p.trace_id,
                "timestamp": p.peak_time.datetime,
                "prob": p.peak_value,
                "type": p.phase.lower()
            })
        pick_df = pd.DataFrame(pick_df)
        

        '''
        with open(os.path.join('/home/javak/.seisbench/datasets/chile/parameters_tunning', 'PhaseNet_result_p_picks.pkl'),'rb') as fp:
            PhaseNet_result_p_picks = pickle.load(fp)
        
        with open(os.path.join('/home/javak/.seisbench/datasets/chile/parameters_tunning', 'PhaseNet_result_s_picks.pkl'),'rb') as ft:
            PhaseNet_result_s_picks = pickle.load(ft)
        
        all_picks = [PhaseNet_result_p_picks, PhaseNet_result_s_picks]
        new_df = pd.concat(all_picks)
        pick_df = new_df[(new_df['timestamp'] > '2012-01-01 00:00:00') & (new_df['timestamp'] < '2012-02-01 00:00:00')]
        '''
        station_df = []
        for station in inv[0]:
            station_df.append({
                "id": f"CX.{station.code}.",
                "longitude": station.longitude,
                "latitude": station.latitude,
                "elevation(m)": station.elevation
            })
        station_df = pd.DataFrame(station_df)

        station_df["x(km)"] = station_df.apply(lambda x: transformer.transform(x["latitude"], x["longitude"])[0] / 1e3, axis=1)
        station_df["y(km)"] = station_df.apply(lambda x: transformer.transform(x["latitude"], x["longitude"])[1] / 1e3, axis=1)
        station_df["z(km)"] = station_df["elevation(m)"] / 1e3

        #northing = {station: y for station, y in zip(station_df["id"], station_df["y(km)"])}
        #station_dict = {station: (x, y) for station, x, y in zip(station_df["id"], station_df["x(km)"], station_df["y(km)"])}
        #pick_df.sort_values("timestamp")

        # Save pick_df file in "/home/javak/.seisbench/datasets/chile/files_paths" directory
        pick_df_name = '{0}{1}{2}{3}{4}'.format('picks_','Pth:', P_th,'_Sth:', S_th)
        pick_df.to_pickle(os.path.join('{0}/{1}/{2}'.format(seisbench.cache_root,'datasets/chile/files_paths',pick_df_name)))

        return station_df, pick_df

    def association (self, pick_df, station_df, config, min_samples, P_th, S_th, min_pk_eq, meth, amp):

        '''
        This function use the following inputs to run association:
            - pick_df (Data Frame): phasenet picks
            - station_df (Data Frame): station information
            - config: configuration of Gamma
        Outputs:
                - catalog (pikle file): this file will be created in the "/home/javak/.seisbench/datasets/chile/files_paths" directory.
                - assignments (pikle file): this file will be created in the "/home/javak/.seisbench/datasets/chile/files_paths" directory.

        '''
        # Run gamma association 
        pbar = tqdm(1)
        catalogs, assignments = association(pick_df, station_df, config, method=config["method"], pbar=pbar)

        catalog = pd.DataFrame(catalogs)
        assignments = pd.DataFrame(assignments, columns=["pick_idx", "event_idx", "prob_gamma"])

        # Save catalog as a pikle in "/home/javak/.seisbench/datasets/chile/files_paths" directory
        catalog_name = '{0}{1}{2}{3}{4}{5}{6}{7}{8}{9}{10}'.format('catalog_','min_pk_eq:', min_pk_eq,'_meth:', meth,'_amp:', amp,'_Pth:', P_th,'_Sth:', S_th,'_min_sample:', min_samples,'_.pkl')
        catalog.to_pickle(os.path.join('{0}/{1}/{2}'.format(seisbench.cache_root,'datasets/chile/files_paths',catalog_name)))

        # Save assignments as a pikle in "/home/javak/.seisbench/datasets/chile/files_paths" directory
        assignments_name = '{0}{1}{2}{3}{4}{5}{6}{7}{8}{9}{10}'.format('assignments_', 'min_pk_eq:', min_pk_eq,'_meth:', meth,'_amp:', amp,'_Pth:', P_th,'_Sth:', S_th,'_min_sample:', min_samples,'_.pkl')
        assignments.to_pickle(os.path.join('{0}/{1}/{2}'.format(seisbench.cache_root,'datasets/chile/files_paths',assignments_name)))

if __name__ == "__main__":

    P_threshold_ls = [0.15]
    S_threshold_ls = [0.3]
    dbscan_min_samples_ls = [3]
    min_picks_per_eq = [5]
    method = ['BGMM']
    use_amp = [False]

    start_year_analysis = 2012
    start_day_analysis = 182
    end_year_analysis = 2012
    end_day_analysis = 182

    obj = Parameter_Tuning (P_threshold_ls = P_threshold_ls, 
                            S_threshold_ls=S_threshold_ls, 
                            dbscan_min_samples_ls= dbscan_min_samples_ls,
                            min_picks_per_eq = min_picks_per_eq,
                            method=method,
                            use_amp=use_amp,
                            start_year_analysis = start_year_analysis,
                            start_day_analysis = start_day_analysis,
                            end_year_analysis = end_year_analysis,
                            end_day_analysis = end_day_analysis)
    dists = obj()

    '''
    P_threshold_ls = [0.5, 0.8, 0.9]
    S_threshold_ls = [0.5, 0.8, 0.9]
    dbscan_min_samples_ls = [3, 4]
    min_picks_per_eq = [5]
    method = ['GMM','BGMM']
    use_amp = [False]
    '''
