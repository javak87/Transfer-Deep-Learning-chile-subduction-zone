import pandas as pd
import numpy as np
import obspy
import os
import pickle
import seisbench

class Amplitude_info (object):
    
     def __init__ (self):
        
        with open(os.path.join('{0}/{1}'.format(seisbench.cache_root,'datasets/chile/files_paths'), "DF_auxiliary_path_file.pkl"),'rb') as fn:
            self.DF_auxiliary_path_file = pickle.load(fn)

    def extract_amplitude (self, p_picks:'pd.DataFrame', s_picks:'pd.DataFrame')-> pd.DataFrame:

        '''This function extract amplitude of corresponding picks captured by phasenet
            parameters:
                        1- p_picks (dataframe): this data frame contains id, timestamp, prob, and type which is
                                                the output of phasenet p_picks.
                        2- s_picks (dataframe): this data frame contains id, timestamp, prob, and type which is
                                                the output of phasenet s_picks.       
            Output:
                    the output data frame contains the information stored in picks data frame plus the three amplitude
                    corresponding N,E, and Z components.
        
        procedure for choosing p pick amplitude:
            1- read the corresponding time of p pick from p_picks data frame
            2- read the corresponding mseed file for three components
            3- take the maximum amplitude between p pick time and the next 5 seconds data for each trace

        procedure for choosing s pick amplitude:
            1- read the corresponding time of s pick from s_picks data frame
            2- read the corresponding mseed file for three components
            3- take the maximum amplitude between s pick time and the next 15 seconds data for each trace.
             in case of appearing a p picks within 15 seconds window, decrease the time window exactly before the p picks.
        '''
        amplitude_p = np.empty([p_picks.shape[0], 4])
        
        # create an numpy array to store p_picks time in numpy array. this file array will be use to
        # select the right interval for choosing S picks amplitude

        p_picks__time_arr = np.zeros((p_picks.shape[0],1))

        print (p_picks.id.iloc[0])
        stream = self.read_data (p_picks.id.iloc[0])
        # create timstamp data in nano second
        timestamp_0 = pd.to_datetime(stream[0].times("timestamp"), unit='s', origin='unix').astype(int) / 10**9
        timestamp_1 = pd.to_datetime(stream[1].times("timestamp"), unit='s', origin='unix').astype(int) / 10**9
        timestamp_2 = pd.to_datetime(stream[2].times("timestamp"), unit='s', origin='unix').astype(int) / 10**9

        # transform start time to UTCDateTime and second and truncate start time after 3 points
        decs = 3
        times_0= np.trunc(timestamp_0*10**decs)/(10**decs)
        times_1= np.trunc(timestamp_1*10**decs)/(10**decs)
        times_2= np.trunc(timestamp_2*10**decs)/(10**decs)

        for i in range (p_picks.shape[0]):

            

            # transform time to UTCDateTime
            p_pick_time = obspy.UTCDateTime(p_picks.timestamp.iloc[i])

            # transform UTCDateTime to second
            p_pick_time = obspy.UTCDateTime.__float__(p_pick_time)

            # store p_pick_time in a numpy array
            p_picks__time_arr[i] = p_pick_time


            # find the index of p_pick_time for three traces
            index_p_0 = np.searchsorted(times_0, p_pick_time)
            index_p_1 = np.searchsorted(times_1, p_pick_time)
            index_p_2 = np.searchsorted(times_2, p_pick_time)
            
            # choosing the right interval and take the maximum amplitude within 5 seconds interval for each trace
            if index_p_0 < times_0.shape[0] - 50:
                amp_p_0 = np.max(np.absolute(stream[0].data[index_p_0:index_p_0+50]), initial=0)
            else:
                amp_p_0 = 0

            if index_p_1 < times_1.shape[0] - 50:
                amp_p_1 = np.max(np.absolute(stream[1].data[index_p_1:index_p_1+50]), initial=0)          
            else:
                amp_p_1 = 0
            
            if index_p_2 < times_2.shape[0] - 50:
                amp_p_2 = np.max(np.absolute(stream[2].data[index_p_2:index_p_2+50]), initial=0)
            else:
                amp_p_2 = 0

            # compute the amplitude
            p_phase_amp = np.sqrt(amp_p_0**2 + amp_p_1**2 + amp_p_2**2)

            amplitude_p[i, 0] =  amp_p_0
            amplitude_p[i, 1] =  amp_p_1
            amplitude_p[i, 2] =  amp_p_2
            amplitude_p[i, 3] =  p_phase_amp

        # store data in p_picks data frame
        p_picks['amp_p_0'] = amplitude_p[:, 0]
        p_picks['amp_p_1'] = amplitude_p[:, 1]
        p_picks['amp_p_2'] = amplitude_p[:, 2]
        p_picks['p_phase_amp'] = amplitude_p[:, 3]


        amplitude_s = np.empty([s_picks.shape[0], 4])
        

        for j in range (s_picks.shape[0]):

           
            # transform time to UTCDateTime
            s_pick_time = obspy.UTCDateTime(s_picks.timestamp.iloc[j])

            # transform UTCDateTime to second
            s_pick_time = obspy.UTCDateTime.__float__(s_pick_time)

            # find the index of s_pick_time for three traces
            index_s_0 = np.searchsorted(times_0, s_pick_time)
            index_s_1 = np.searchsorted(times_1, s_pick_time)
            index_s_2 = np.searchsorted(times_2, s_pick_time)
            

            # choosing the right interval to select amplitude


            if index_s_0 < times_0.shape[0] - 150:
                time_0_check = times_0[index_s_0+150]
                check_tr_0 = p_picks__time_arr[np.where((p_picks__time_arr>s_pick_time) & (p_picks__time_arr<time_0_check))]
                
                if check_tr_0.shape[0] !=0:
                    inx_0 = np.searchsorted(times_0, check_tr_0)
                    amp_s_0 = np.max(np.absolute(stream[0].data[index_s_0:inx_0[0]]),initial=0)
                else:
                    amp_s_0 = np.max(np.absolute(stream[0].data[index_s_0:index_s_0+150]), initial=0)

            else:
                amp_s_0 = 0


            if index_s_1 < times_1.shape[0] - 150:
                time_1_check = times_1[index_s_1+150]
                check_tr_1 = p_picks__time_arr[np.where((p_picks__time_arr>s_pick_time) & (p_picks__time_arr<time_1_check))]
                
                if check_tr_1.shape[0] !=0:
                    inx_1 = np.searchsorted(times_1, check_tr_1)
                    amp_s_1 = np.max(np.absolute(stream[1].data[index_s_1:inx_1[0]]),initial=0)
                else:
                    amp_s_1 = np.max(np.absolute(stream[1].data[index_s_1:index_s_1+150]),initial=0)
            else:
                amp_s_1 = 0


            if index_s_2 < times_2.shape[0] - 150:
                time_2_check = times_2[index_s_2+150]
                check_tr_2 = p_picks__time_arr[np.where((p_picks__time_arr>s_pick_time) & (p_picks__time_arr<time_2_check))]
                
                if check_tr_2.shape[0] !=0:
                    inx_2 = np.searchsorted(times_2, check_tr_2)
                    amp_s_2 = np.max(np.absolute(stream[2].data[index_s_2:inx_2[0]]),initial=0)
                else:
                    amp_s_2 = np.max(np.absolute(stream[2].data[index_s_2:index_s_2+150]),initial=0)
            else:
                amp_s_2 = 0

            s_phase_amp = np.sqrt(amp_s_0**2 + amp_s_1**2 + amp_s_2**2)

            amplitude_s[j, 0] =  amp_s_0
            amplitude_s[j, 1] =  amp_s_1
            amplitude_s[j, 2] =  amp_s_2
            amplitude_s[j, 3] =  s_phase_amp

        s_picks['amp_s_0'] = amplitude_s[:, 0]
        s_picks['amp_s_1'] = amplitude_s[:, 1]
        s_picks['amp_s_2'] = amplitude_s[:, 2]
        s_picks['s_phase_amp'] = amplitude_s[:, 3]

        return p_picks, s_picks

    
    def read_data (self, daily_data:'str') -> obspy:

        '''
        Read the mseed daily data and return stream.
            Parameters:
                - daily_data (str): The name of daily mseed file ( like CX.PB06..HHZ.D.2020.366)
            return:
                    - stream: obspy stream data
        '''
        stream = obspy.read(os.path.join(self.working_direc, 'mseed', '{0}'.format(daily_data)), sep="\t")
        return stream