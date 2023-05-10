import numpy as np
import pandas as pd
import pickle

file_name = '/home/javak/Sample_data_chile/Events_catalog/Manual picks/Vaclav/Manual/picks_2010_058'

names=['0','1','2','3','4','5','6','7','phase_hint','9','10']
df = pd.read_csv(file_name,names=names, header=0,on_bad_lines='skip', delim_whitespace=True)
df.drop(columns=['0','3','5','6','7','9','10'], inplace=True)
df["picks_time"] = df["1"] + 'T' + df["2"] + '00Z'
df[['network_code', 'station_code', 'point','ch']] = df['4'].str.split(".", expand = True)
df.drop(columns=['1','2','4','point','ch'], inplace=True)
df = df[['picks_time', 'network_code', 'station_code', 'phase_hint']]
df.to_pickle('/home/javak/Sample_data_chile/Events_catalog/Manual picks/Vaclav/Manual/picks_2010_058.pkl')