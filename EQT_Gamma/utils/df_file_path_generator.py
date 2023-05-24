import pandas as pd
import numpy as np
import os
import pickle

def generate_DF_file_path (chile_GFZ_online_traj, export_fig_path):

    data =[]
    
    for path,subdir,files in os.walk(chile_GFZ_online_traj):
        #for name_dir in subdir:
            #folder = os.path.join(path,name_dir) # will print path of directories
  
        for file_name in files:
            if  file_name.startswith('.') == False: 
                path_name = os.path.join(path,file_name) # will print path of files
                data.append((path_name, file_name))
    # creat DataFrame
    df = pd.DataFrame(data, columns=['path', 'file_name'])
    df = df.sort_values(by='path', ascending=True)
    #df = df.drop([81313,81717])
    # Split the file_name column
    days_df = df['file_name'].str.split('.', -1, expand=True)
    days_df = days_df.rename(columns = {0:'network',1:'station',3:'stream_component',5:'year',6:'day'})
    days_df = days_df.drop(columns=[2, 4])

    # convert year and day to int
    days_df.year = pd.to_numeric(days_df.year)
    days_df.day = pd.to_numeric(days_df.day)
    # Combine two DataFrame
    DF_file_path = pd.concat([df, days_df], axis=1)

    # Save DataFrame
    file_name = 'DF_chile_path_file.pkl'
    DF_file_path.to_pickle(os.path.join(export_fig_path , file_name))



if __name__ == "__main__":

    chile_GFZ_online_traj = '/data2/chile/CHILE_COMBINED_2021'
    export_fig_path = '/home/javak/Sample_data_chile/Comparing PhaseNet and Catalog'
    start_year_analysis = 2006
    start_day_analysis = 1
    end_year_analysis = 2021
    end_day_analysis = 365

    generate_DF_file_path (chile_GFZ_online_traj,export_fig_path)
