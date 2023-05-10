import pandas as pd
import pickle
from obspy import Catalog
from obspy.core.event.event import Event
from obspy.core.event.origin import Origin
from obspy.core.event.origin import Pick
from obspy.core.event.base import WaveformStreamID
from obspy import UTCDateTime
from obspy.core.event import Origin
from obspy.clients.fdsn import Client



#picks_name = '/home/javak/Sample_data_chile/Comparing PhaseNet and Catalog/EQTransformer_transfer learning_instance/Binary entropy/2011.90_(Train_on_nosshin_dataset/P=0,22, s= 0.12/PhaseNet_result_p&s.pkl'
#picks_name = '/home/javak/Sample_data_chile/Events_catalog/Manual picks/Jonas/picks_2011_090_cleaned.pkl'
picks_name = '/home/javak/Sample_data_chile/Events_catalog/Manual picks/Cristian_Jonas_Nooshin_manual_picks/Cristian_Jonas_Nooshin_manual_picks.pkl'
autopicker_out = False

with open (picks_name, 'rb') as pk:
    picks = pickle.load(pk)

if autopicker_out == True:
    picks[['network_code', 'others']] = picks['id'].str.split('.', 1, expand=True)
    picks[['station_code', 'dummpy']] = picks['others'].str.split('.', 1, expand=True)
    picks.drop(['dummpy','others'], axis=1, inplace=True)
    picks['type'] = picks['type'].apply(str.upper)
else:
    picks.rename(columns={"picks_time": "timestamp", "phase_hint": "type"},inplace=True)
    picks['id'] = picks['network_code'] + '.' + picks['station_code'] + '.'

starttime = UTCDateTime(picks['timestamp'].iloc[0])
endtime = UTCDateTime(picks['timestamp'].iloc[-1])
client = Client("GFZ")
inv = client.get_stations(network="CX", station="*", 
                location="*", channel="HH?",starttime=starttime,endtime=endtime)

picks_event = pd.DataFrame()
for i in range(len(inv[0])):
    station_pick = picks[picks['station_code'] == inv[0].stations[i].code]

    if station_pick.shape[0] == 0:
        continue
    
    station_pick.sort_values(by=['timestamp'], inplace =True)

    #start_time = UTCDateTime(station_pick['timestamp'].iloc[0])
    
    
    station_pick['UTC_time'] = station_pick['timestamp'].apply(lambda x: UTCDateTime(x))

    #while start_time < station_pick['timestamp'].iloc[-1]:
    #p_picks = station_pick[station_pick['type'] == 'P']
    for line in station_pick.iterrows():

        if line[1].type == 'S':
            continue
        
        start_time = UTCDateTime(line[1].timestamp)
        end_time = start_time + 60

        picks_min_inter = station_pick[(station_pick['UTC_time'] >=start_time) & (station_pick['UTC_time'] < end_time)]

        if picks_min_inter.shape[0] < 2:
             continue
        
        condition_0 = (picks_min_inter['type'].iloc[0] == 'P' and picks_min_inter['type'].iloc[1] =='S')
        phase_dist = UTCDateTime(picks_min_inter['timestamp'].iloc[1]) - UTCDateTime(picks_min_inter['timestamp'].iloc[0])
        if (condition_0) and (phase_dist > 5):
            picks_event = pd.concat([picks_event, picks_min_inter.iloc[0:2]], axis=0)
            
        else:
            continue

picks_event.drop(['UTC_time'], axis =1, inplace=True)

cat = Catalog()
ev = Event()

for i in range(picks_event.shape[0]):

    stream_id = WaveformStreamID(
                    network_code = picks_event['network_code'].iloc[i],
                    station_code = picks_event['station_code'].iloc[i],
                    channel_code = 'HHZ')
    pick = Pick(
            time=UTCDateTime(picks_event['timestamp'].iloc[i]),
            waveform_id=stream_id,
            phase_hint=picks_event['type'].iloc[i])

    ev.picks += [pick]
    
    
    if i % 2 == 1 and i >0:
        origin = Origin()
        origin.time = UTCDateTime(ev.picks[0].time)
        origin.time_errors.uncertainty = 0.01
        origin.latitude = 12
        origin.latitude_errors.uncertainty = 0.01
        origin.longitude = 42
        origin.longitude_errors.uncertainty = 0.01
        origin.depth = 50000
        origin.depth_errors.uncertainty = 0.1
        origin.depth_type = 'from location'
        ev.origins += [origin]
        cat.events += [ev]
        ev = Event()
    
#cat.events += [ev]
cat.write("/home/javak/Sample_data_chile/Events_catalog/Manual picks/Cristian_Jonas_Nooshin_manual_picks/dummy catalog/Cristian_Jonas_Nooshin_catalog.xml", format="QUAKEML") 
v = 1


