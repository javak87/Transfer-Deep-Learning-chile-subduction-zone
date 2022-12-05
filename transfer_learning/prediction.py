import math
import os
import pickle

import numpy as np
import obspy
import pandas as pd
import seisbench
import seisbench.data as sbd
import seisbench.generate as sbg
import seisbench.models as sbm
import torch
import torch.nn as nn
import torch.optim as optim


class Data_Augmentation(object):

    def __init__ (self, model_dir, 
                start_year_analysis, start_day_analysis,
                end_year_analysis, end_day_analysis):
        
   
        #self.augmentations = augmentations
        self.model_dir = model_dir
        self.start_year_analysis = start_year_analysis
        self.start_day_analysis = start_day_analysis
        self.end_year_analysis = end_year_analysis
        self.end_day_analysis = end_day_analysis
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __call__ (self, p_th=0.1, s_th=0.6):

        DF_selected_chile_path_file, DF_auxiliary_path_file = self.waveforme_data_frame()
        
        df = pd.DataFrame(columns=['id', 'timestamp', 'type'])
        
        for stream in self.waveforme_loader (DF_selected_chile_path_file, DF_auxiliary_path_file):
            for stream in self.sliding_window_loader (stream):
                id  = stream[0].id[0:np.where(np.array(list(stream[0].id)) == '.')[0][1]+1]
                base_time = stream[0].stats.starttime
                data = self.augmentation(stream)
                output = self.predict (data,p_th, s_th)
                p_index, s_index = self.extract_p_s_index(output)
                p_index = self.extract_central_pick(p_index)
                s_index = self.extract_central_pick(s_index)

                if len(p_index) !=0:
                    #p_index = self.extract_central_pick(p_index)
                    p_frame = self.extract_p_picks(p_index, base_time,id)
                    new_frame_p = pd.DataFrame(p_frame, columns=['id', 'timestamp', 'type'])
                    df = pd.concat([df, new_frame_p])

                if len(s_index) !=0:
                    #s_index = self.extract_central_pick(s_index)
                    s_frame= self.extract_s_picks(s_index, base_time,id)
                    new_frame_s = pd.DataFrame(s_frame, columns=['id', 'timestamp', 'type'])
                    df = pd.concat([df, new_frame_s])
                            
            df.to_csv('/home/javak/Transfer-Deep-Learning-chile-subduction-zone/transfer_learning/df.csv')
            nnn
    def extract_central_pick(self, index):
        if index.reshape(1,-1).shape[1]>23:
            num_array = np.zeros((index.shape[0]-1,2))
            index = index.cpu().detach().numpy()
            num_array[:,0] = np.diff(index)
            num_array[:,1] = index[1:]
            split = np.where(num_array[:,0]!=1)[0].tolist()
            if len(split) !=0:
                group = np.split(num_array, split, axis=0)
                index = np.floor([np.median(sample[1:][:,1]) for sample in group if sample.shape[0]>20]).astype(int).tolist()
            else:
                index = [np.floor(np.median(index)).astype(int)]
        else:
            index=[]
        return index
    
    def extract_p_picks(self, p_index, base_time,id):
        p_picks = [(base_time + 0.01*index).strftime("%Y-%m-%d %H:%M:%S.%f") for index in p_index]
        return zip(len(p_picks)*[id],p_picks,len(p_picks)*['p'])

    def extract_s_picks(self, s_index, base_time,id):
        s_picks = [(base_time + 0.01*index).strftime("%Y-%m-%d %H:%M:%S.%f") for index in s_index]
        return zip(len(s_picks)*[id],s_picks,len(s_picks)*['s'])

    def extract_p_s_index(self, output):
        p_index = (output[:,0,:].view(-1)==1).nonzero().squeeze()
        s_index = (output[:,1,:].view(-1)==1).nonzero().squeeze()
        return p_index, s_index

    def model_loader (self):
        
        # load model
        model = sbm.PhaseNet.from_pretrained('instance').to(device="cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(self.model_dir)
        model.load_state_dict(checkpoint)
        return model
    def augmentation(self, stream):

        data = np.zeros((3,3001))
        data[0,:] = (stream[0].data - np.mean(stream[0].data)) / np.std(stream[0].data)
        data[1,:] = (stream[1].data - np.mean(stream[1].data)) / np.std(stream[1].data)
        data[2,:] = (stream[2].data - np.mean(stream[2].data)) / np.std(stream[2].data)
        return data

    def waveforme_data_frame (self):

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

        DF_selected_chile_path_file = chile_path_file.drop_duplicates()

        # creat new DataFrame to make sure all 3-components are existed
        df_counter = chile_path_file.groupby(['network','station', 'year', 'day']).size().reset_index(name='count')
        df_counter = df_counter[df_counter['count']==3]

        # drop the 'count' column
        df_counter = df_counter.drop(columns=['count'])
        DF_auxiliary_path_file = df_counter.sort_values(by=['year', 'day'])

        return DF_selected_chile_path_file, DF_auxiliary_path_file
    
    def waveforme_loader (self, DF_selected_chile_path_file, DF_auxiliary_path_file):
            
            i = 0

            while i <= DF_auxiliary_path_file.shape[0]:

                df = DF_selected_chile_path_file[['network', 'station', 'year','day']]==DF_auxiliary_path_file.iloc[i]
                df_components = DF_selected_chile_path_file[(df['network']== True) & 
                                (df['station']== True) & (df['year']== True) &
                                (df['day']== True)]

                trZ = obspy.read(df_components.path.iloc[2])
                trN = obspy.read(df_components.path.iloc[1])
                trE = obspy.read(df_components.path.iloc[0])


                trN.append(trE[0])
                trN.append(trZ[0])
                trN = trN.sort()
                yield trN

                i += 1
    
    def sliding_window_loader (self, stream):
         
        # check common time interval
        start_list = [float(stream[0].stats.starttime), float(stream[1].stats.starttime), float(stream[2].stats.starttime)]
        index_start =start_list.index(max(start_list))

        end_list = [float(stream[0].stats.endtime), float(stream[1].stats.endtime), float(stream[2].stats.endtime)]
        index_end =end_list.index(min(end_list))

        stream = stream.trim(stream[index_start].stats.starttime, stream[index_end].stats.endtime)

        i = 0
        first_point = stream[0].stats.starttime
        last_point = stream[0].stats.starttime + 30.00
        while  i<= stream[0].data.shape[0]-3002: 
            #st0 = stream[0].data[i:j].reshape(1,stream[0].data[i:j].shape[0])
            #st1 = stream[1].data[i:j].reshape(1,stream[1].data[i:j].shape[0])
            #st2 = stream[2].data[i:j].reshape(1,stream[2].data[i:j].shape[0])
            #data = np.concatenate((st0, st1, st2), axis=0)
            stra = stream.slice(first_point,last_point)
            yield stra
            first_point = last_point - 0.5
            last_point = last_point + 30.00 - 0.5
            i += 3001
            #j = i + in_samples

    def predict (self, data, p_th, s_th):

        model = self.model_loader ()
        model.eval()
        input = torch.tensor(data, dtype=torch.float32).to(self.device).reshape(1,3,3001)
        with torch.no_grad():
            
            output = model(input)
            output[:,0,:] = torch.where(output[:,0,:] > p_th, 1 , 0)
            output[:,1,:] = torch.where(output[:,1,:] > s_th, 1 , 0)
            output[:,2,:] = 1 - (output[:,0,:] + output[:,1,:])
        
        return output
            


if __name__ == '__main__':


    start_year_analysis = 2012
    start_day_analysis = 182
    end_year_analysis = 2012
    end_day_analysis = 182

    #stream = 5
    
    model_dir = '/home/javak/Transfer-Deep-Learning-chile-subduction-zone/transfer_learning/transfer_learing_phasenet.pth.tar'

    aug_obj = Data_Augmentation(model_dir, start_year_analysis,start_day_analysis,
                                end_year_analysis, end_day_analysis)
    
    aug_obj()
    #DF_selected_chile_path_file, DF_auxiliary_path_file = aug_obj.waveforme_data_frame()
    
    #aug_obj.sliding_window_loader(stream)
