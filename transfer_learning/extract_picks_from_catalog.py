import seisbench
import seisbench.data as sbd
import seisbench.util as sbu

import obspy
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNException
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import os
import pandas as pd

first_day = '/home/javak/Sample_data_chile/Events_catalog/Updated Catalog from Crist/catalog_2007_2021/manual picks/day1.xml'
client = Client("GFZ", timeout=10)
catalog_day_1 = obspy.core.event.read_events(pathname_or_url=first_day, format=None)
second_day = '/home/javak/Sample_data_chile/Events_catalog/Updated Catalog from Crist/catalog_2007_2021/manual picks/day2.xml'
catalog_day_2 = obspy.core.event.read_events(pathname_or_url=second_day, format=None)
catalog = catalog_day_1.__add__(catalog_day_2)

picks = pd.DataFrame(columns={'timestamp','network_code','station_code','phase_hint'})

for i in range (len(catalog.events)):
    event = catalog.events[i]


    for j in range (len(event.picks)):
        time = event.picks[j].time
        network = event.picks[j].waveform_id.network_code
        station = event.picks[j].waveform_id.station_code
        #id = event.picks[j].waveform_id.id[0:-4]
        phase_hint = event.picks[j].phase_hint
        new_picks = {'timestamp':[str(time)],'network_code':[network],'station_code':[station],'phase_hint':[phase_hint]}
        data_frame= pd.DataFrame(data=new_picks)
        picks = pd.concat([picks,data_frame], axis=0)

s_picks = picks[picks.phase_hint =='S']
p_picks = picks[picks.phase_hint =='P']
picks.to_pickle(os.path.join('/home/javak/Sample_data_chile/Events_catalog/Updated Catalog from Crist/catalog_2007_2021/manual picks/Extract_picks_from_catalog', 'Nooshin_picks.pkl'))
s_picks.to_pickle(os.path.join('/home/javak/Sample_data_chile/Events_catalog/Updated Catalog from Crist/catalog_2007_2021/manual picks/Extract_picks_from_catalog', 'Nooshin_s_picks.pkl'))
p_picks.to_pickle(os.path.join('/home/javak/Sample_data_chile/Events_catalog/Updated Catalog from Crist/catalog_2007_2021/manual picks/Extract_picks_from_catalog', 'Nooshin_p_picks.pkl'))
