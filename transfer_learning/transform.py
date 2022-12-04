import copy

import numpy as np
import obspy
from obspy import UTCDateTime
from obspy.clients.fdsn import Client


class Trace_Transform(object):

    def __init__(self, waveform, z_rotat=True, n_rotat=True, w_rotat=True):

        self.waveform = copy.deepcopy(waveform)
        self.z_rotat = z_rotat
        self.n_rotat = n_rotat
        self.w_rotat =w_rotat

    def _chanel_order (self):
        try:
            self.waveform.traces[0].id = 'Z'
            self.waveform.traces[0].id = 'E'
            self.waveform.traces[0].id = 'N'
        except ValueError:
            print('the channel order is diffrent from setting')

    def z_rotation(self, angle_deg:'float'):

        angle_rad = np.array((angle_deg)) * np.pi / 180.
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                                [np.sin(angle_rad), np.cos(angle_rad), 0],
                                [0,0,1]])
        
        trace_0 = self.waveform.traces[0].data.reshape(self.waveform.traces[0].data.shape[0],1)
        trace_1 = self.waveform.traces[1].data.reshape(self.waveform.traces[1].data.shape[0],1)
        trace_2 = self.waveform.traces[2].data.reshape(self.waveform.traces[1].data.shape[0],1)
        
        stacked_straem = np.hstack((np.hstack((trace_0,trace_1)),trace_2))
        rotated_stream = np.matmul(stacked_straem, rotation_matrix)
        
        self.waveform.traces[0] = obspy.core.trace.Trace(rotated_stream[:,0])
        self.waveform.traces[1] = obspy.core.trace.Trace(rotated_stream[:,1])
        self.waveform.traces[2] = obspy.core.trace.Trace(rotated_stream[:,2])
        
        return self.waveform

if __name__ == '__main__':
    
    client = Client("GFZ")
    t = UTCDateTime("2012/06/30 00:08:58")
    stream = client.get_waveforms(network="CX", station="PB01", location="*", channel="HH?", starttime=t, endtime=t+10)

    obj =Trace_Transform(stream).z_rotation(30.)
    v= 10