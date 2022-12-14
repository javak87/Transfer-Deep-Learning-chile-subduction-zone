import time

from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import matplotlib.pyplot as plt
from obspy.signal.cross_correlation import correlate
import scipy
import numpy as np

client = Client("GFZ")

t = UTCDateTime("2007/01/02 05:48:50")
stream_PB01 = client.get_waveforms(network="CX", station="PB01", location="*", channel="HH?", starttime=t+9, endtime=t+10)
stream_PB02 = client.get_waveforms(network="CX", station="PB02", location="*", channel="HH?", starttime=t+2, endtime=t+50)
tim1_obs = time.time()
cc =correlate(stream_PB01[0], stream_PB02[0], stream_PB02[0].data.shape[0]-stream_PB01[0].data.shape[0])
print(np.argmax(cc))
tim2_obs = time.time()
print('obpsy cross corr time: ', tim2_obs-tim1_obs)

sig2 = stream_PB02[0].data.tolist()
sig1 = stream_PB01[0].data.tolist()

# Pre-allocate correlation array
'''
corr = abs(len(sig1) - len(sig2)  + 1) * [0]
tim1_py = time.time()
# Go through lag components one-by-one
for l in range(len(corr)):
    corr[l] = sum([sig1[i+l] * sig2[i] for i in range(len(sig2))])
tim2_py = time.time()
print('python cross corr time: ', tim2_py-tim1_py)
'''
# using scipy
tim1_sp_same = time.time()
corr_sp = scipy.signal.correlate(sig1, sig2, mode='same')
print(np.argmax(corr_sp))
tim2_sp_same = time.time()
print('scipy cross corr time: ', tim2_sp_same-tim1_sp_same)
#### using Fourie Trasform for cross correlation
tim1_sp_fft = time.time()
corr_sp = scipy.signal.correlate(sig1, sig2, mode='same', method='fft')
print(np.argmax(corr_sp))
tim2_sp_fft = time.time()
print('scipy cross corr time fft: ', tim2_sp_fft-tim1_sp_fft)

sig1= np.array(sig1)
sig2= np.array(sig2)
sig1_pad = np.pad(sig1, (0, sig2.shape[0]-sig1.shape[0]), 'constant')
#sig2_pad = np.pad(sig2, (0, sig1.shape[0]+1), 'constant')
sig2_pad = sig2

fft1_sig1 = np.fft.fft(sig1_pad).reshape(1,sig1_pad.shape[0])
fft1_sig2 = np.fft.fft(sig2_pad).reshape(1,sig2_pad.shape[0])
cross_power_spectrum = (fft1_sig1 * fft1_sig2.conj()) / np.abs(fft1_sig1* fft1_sig2.conj())
r = np.abs(np.fft.ifft2(cross_power_spectrum))


