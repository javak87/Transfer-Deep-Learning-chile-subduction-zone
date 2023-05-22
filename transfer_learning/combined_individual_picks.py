import numpy as np
import pandas as pd
import pickle
import os



directory = '/home/javak/Sample_data_chile/Comparing PhaseNet and Catalog/EQT_Instance_CJN/p=0.09_s=0.1_d=0.005'
picker_name = '2017_217_p=0.09_s=0.1_d=0.005'
picks_name =[]
for path,subdir,files in os.walk(directory):
    #for name_dir in subdir:
        #folder = os.path.join(path,name_dir) # will print path of directories

    for file_name in files:
        if  file_name.startswith('.') == False: 
            path_name = os.path.join(path,file_name) # will print path of files
            #data.append((path_name))
            print(path_name)
        with open(path_name,'rb') as fp:
            new_file = pickle.load(fp)
            print(new_file.shape)
        if len(picks_name) == 0:
            picks_name = new_file
            continue
        picks_name = pd.concat([picks_name,new_file ],axis=0)

picks_name.to_pickle('{0}/{1}.{2}'.format(directory, picker_name,"pkl"))

with open('{0}/{1}.{2}'.format(directory, picker_name,"pkl"),'rb') as fp:
    file = pickle.load(fp)
v = 0