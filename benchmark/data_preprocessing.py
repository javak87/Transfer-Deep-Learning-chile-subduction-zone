import pandas as pd
import obspy
import pickle
import os
from os import path
import seisbench

class Data_Preprocessing(object):

    def __init__(self, start_year_analysis, start_day_analysis, 
                    end_year_analysis, end_day_analysis):

        '''
        This function select and store streams with the given interval
        '''
        self.start_year_analysis = start_year_analysis
        self.start_day_analysis = start_day_analysis
        self.end_year_analysis = end_year_analysis
        self.end_day_analysis = end_day_analysis

        #self.export_DF_path = '{0}/{1}'.format(seisbench.cache_root,'datasets/chile/files_paths')
    
      

    def DF_path (self):

        '''
        This function loads "DF_chile_path_file.pkl" file (path of all mseed files
        from export_DF_path and filter the data between the given interval.
        After using this function, 'DF_selected_chile_path_file.pkl' and 
        'DF_auxiliary_path_file.pkl' will be created.

        Important note: This function will not consider mseed file with less than 3- components.

        Input:
            - "DF_chile_path_file.pkl"

        Outputs: Exports two data frame files in the '{0}/{1}'.format(seisbench.cache_root,'datasets/chile/files_paths') directory.
            - DF_selected_chile_path_file.pkl
            - DF_auxiliary_path_file.pkl
        '''

        # Read pickle data (Path of all chile stream data)
        with open(os.path.join('{0}/{1}'.format(seisbench.cache_root,'datasets/chile/files_paths'), "DF_chile_path_file.pkl"),'rb') as fp:
            chile_path_file = pickle.load(fp)


        chile_path_file = chile_path_file[
                    (chile_path_file['year']>= self.start_year_analysis) 
                    & (chile_path_file['year']<= self.end_year_analysis)]

        chile_path_file['convert_yeartoday']= 365*chile_path_file['year']+chile_path_file['day']
        
        # creat upper and lower limit to filter
        lower_limit = 365*self.start_year_analysis + self.start_day_analysis 
        upper_limit = 365*self.end_year_analysis   + self.end_day_analysis

        # Apply filter
        chile_path_file = chile_path_file[(chile_path_file['convert_yeartoday']>= lower_limit) & 
                (chile_path_file['convert_yeartoday']<= upper_limit)]   

        chile_path_file = chile_path_file.drop_duplicates()

        # creat new DataFrame to make sure all 3-components are existed
        df_counter = chile_path_file.groupby(['network','station', 'year', 'day']).size().reset_index(name='count')
        df_counter = df_counter[df_counter['count']==3]

        # drop the 'count' column
        df_counter = df_counter.drop(columns=['count'])
        df_counter = df_counter.sort_values(by=['year', 'day'])
        # Save selected DataFrame based on given time interval
        chile_path_file.to_pickle(os.path.join('{0}/{1}'.format(seisbench.cache_root,'datasets/chile/files_paths') , 'DF_selected_chile_path_file.pkl'))

        # Save auxiliary DataFrame based on given time interval
        df_counter.to_pickle(os.path.join('{0}/{1}'.format(seisbench.cache_root,'datasets/chile/files_paths') , 'DF_auxiliary_path_file.pkl'))

    def get_waveforms_chile(self):
        
        '''
        This functions read "DF_auxiliary_path_file.pkl" and "DF_chile_path_file.pkl" files
            and creates one stream.
                    
        '''
        existed_path_file = path.exists(os.path.join('{0}/{1}'.format(seisbench.cache_root,'datasets/chile/files_paths'), "DF_chile_path_file.pkl"))

        
        if existed_path_file == True:
            self.DF_path()

        stream = []

        with open(os.path.join('{0}/{1}'.format(seisbench.cache_root,'datasets/chile/files_paths'), "DF_auxiliary_path_file.pkl"),'rb') as fn:
            DF_auxiliary_path_file = pickle.load(fn)
        
        with open(os.path.join('{0}/{1}'.format(seisbench.cache_root,'datasets/chile/files_paths'), "DF_selected_chile_path_file.pkl"),'rb') as fs:
            DF_selected_chile_path_file = pickle.load(fs)


        df = DF_selected_chile_path_file[['network', 'station', 'year','day']]==DF_auxiliary_path_file.iloc[0]
        df_components = DF_selected_chile_path_file[(df['network']== True) & 
                        (df['station']== True) & (df['year']== True) &
                        (df['day']== True)]

        trZ = obspy.read(df_components.path.iloc[2])
        trN = obspy.read(df_components.path.iloc[1])
        trE = obspy.read(df_components.path.iloc[0])

        trN.append(trE[0])
        trN.append(trZ[0])
        stream = trN.sort()

        for i in range (1, DF_auxiliary_path_file.shape[0]):

            
            # Apply filter to determine the three components in "DF_selected_chile_path_file"
            df = DF_selected_chile_path_file[['network', 'station', 'year','day']]==DF_auxiliary_path_file.iloc[i]
            df_components = DF_selected_chile_path_file[(df['network']== True) & 
                            (df['station']== True) & (df['year']== True) &
                            (df['day']== True)]

            # Read three components mseed file 
            #one_st_stream = []

            trZ = obspy.read(df_components.path.iloc[2])
            trN = obspy.read(df_components.path.iloc[1])
            trE = obspy.read(df_components.path.iloc[0])

            #streamN += streamE
            #streamN += streamZ
            #st = streamN.sort()

            trN.append(trE[0])
            trN.append(trZ[0])
            trN = trN.sort()

            stream.append(trN[0])
            stream.append(trN[1])
            stream.append(trN[2])
            #one_st_stream += streamN
            #one_st_stream += streamE
            #one_st_stream += streamZ
            #one_st_stream = one_st_stream.sort()

            #stream += one_st_stream
            


            #stream.append(mini_stream)

        return stream