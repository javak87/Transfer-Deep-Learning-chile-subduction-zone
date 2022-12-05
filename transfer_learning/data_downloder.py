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


class Data_Downloder (object):
    def __init__ (self, catalog_name):
        self.catalog_name = catalog_name

        base_path = Path(".")
        self.metadata_path = base_path / "metadata.csv"
        self.waveforms_path = base_path / "waveforms.hdf5"
        self.catalog = obspy.core.event.read_events(pathname_or_url=self.catalog_name, format=None)
  
    def __call__ (self):

        self.convert_trace ()

    def get_event_params(self, event):
        origin = event.origins[0]
        #mag = event.preferred_magnitude()

        source_id = str(event.origins[0].resource_id)

        event_params = {
            "source_id": source_id,
            "source_origin_time": str(origin.time),
            "source_origin_uncertainty_sec": origin.time_errors["uncertainty"],
            "source_latitude_deg": origin.latitude,
            "source_latitude_uncertainty_km": origin.latitude_errors["uncertainty"],
            "source_longitude_deg": origin.longitude,
            "source_longitude_uncertainty_km": origin.longitude_errors["uncertainty"],
            "source_depth_km": origin.depth / 1e3,
            "source_depth_uncertainty_km": origin.depth_errors["uncertainty"] / 1e3,
        }

        #if mag is not None:
        #    event_params["source_magnitude"] = mag.mag
        #    event_params["source_magnitude_uncertainty"] = mag.mag_errors["uncertainty"]
        #    event_params["source_magnitude_type"] = mag.magnitude_type
        #    event_params["source_magnitude_author"] = mag.creation_info.agency_id
        
        #if str(origin.time) < "2015-01-07":
        #    split = "train"
        #elif str(origin.time) < "2015-01-08":
        #    split = "dev"
        #else:
        #    split = "test"
        split = 'train'
        event_params["split"] = split
        
        return event_params
    
    def get_trace_params(self, pick):
        net = pick.waveform_id.network_code
        sta = pick.waveform_id.station_code

        trace_params = {
            "station_network_code": net,
            "station_code": sta,
            "trace_channel": pick.waveform_id.channel_code[:2],
            "station_location_code": pick.waveform_id.location_code,
        }

        return trace_params

    def get_waveforms(self, pick, trace_params, time_before=30, time_after=30):
        client = Client("GFZ", timeout=10)

        t_start = pick.time - time_before
        t_end = pick.time + time_after
        
        try:
            waveforms = client.get_waveforms(
                network=trace_params["station_network_code"],
                station=trace_params["station_code"],
                location="*",
                channel=f"{trace_params['trace_channel']}*",
                starttime=t_start,
                endtime=t_end,
            )
        except FDSNException:
            # Return empty stream
            waveforms = obspy.Stream()
        
        return waveforms

    
    def convert_trace (self):


        # Iterate over events and picks, write to SeisBench format
        with sbd.WaveformDataWriter(self.metadata_path, self.waveforms_path) as writer:
            
            # Define data format
            writer.data_format = {
                "dimension_order": "CW",
                "component_order": "ZNE",
                "measurement": "velocity",
                "unit": "counts",
                "instrument_response": "not restituted",
            }
            
            for event in self.catalog:
                event_params = self.get_event_params(event)
                for pick in event.picks:
                    trace_params = self.get_trace_params(pick)
                    waveforms = self.get_waveforms(pick, trace_params)
                    
                    if len(waveforms) == 0:
                        # No waveform data available
                        continue
                
                    sampling_rate = waveforms[0].stats.sampling_rate
                    # Check that the traces have the same sampling rate
                    assert all(trace.stats.sampling_rate == sampling_rate for trace in waveforms)
                    
                    actual_t_start, data, _ = sbu.stream_to_array(
                        waveforms,
                        component_order=writer.data_format["component_order"],
                    )
                    
                    trace_params["trace_sampling_rate_hz"] = sampling_rate
                    trace_params["trace_start_time"] = str(actual_t_start)
                    
                    sample = (pick.time - actual_t_start) * sampling_rate
                    trace_params[f"trace_{pick.phase_hint}_arrival_sample"] = int(sample)
                    trace_params[f"trace_{pick.phase_hint}_status"] = pick.evaluation_mode
                    
                    writer.add_trace({**event_params, **trace_params}, data)

if __name__ == '__main__':

    catalog_name = "/home/javak/Sample_data_chile/Updated Catalog from Crist/catalog_2007_2021/manual picks/catalog_2012_2018.xml"
    obj = Data_Downloder (catalog_name)
    obj()
