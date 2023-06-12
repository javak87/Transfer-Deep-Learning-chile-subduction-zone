import matplotlib.pyplot as plt
import os
import numpy as np
import obspy

class Visualization():
    def __init__(self) :

        if os.path.exists(os.getcwd() + '/result/'):
            pass
        else:
            os.mkdir(os.getcwd() + '/result/')

        if os.path.exists('./result/plots'):
            pass
        else:
            os.mkdir('./result/plots')

    def plot_catalog (self, catalog, station_df ):

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.set_aspect("equal")
        cb = ax.scatter(catalog["x(km)"], catalog["y(km)"], c=catalog["z(km)"], s=8, cmap="viridis")
        cbar = fig.colorbar(cb)
        cbar.ax.set_ylim(cbar.ax.get_ylim()[::-1])
        cbar.set_label("Depth[km]")

        ax.plot(station_df["x(km)"], station_df["y(km)"], "r^", ms=10, mew=1, mec="k")
        ax.set_xlabel("Easting [km]")
        ax.set_ylabel("Northing [km]")

        file_name = '{0}.{extention}'.format('./result/plots/catalog', extention='png')
        fig.savefig(file_name, facecolor = 'w')

    def plot_waveform (self, picks, catalog, assignments, stream, station_dict):

        event_idx = np.random.randint(len(catalog))
        event_picks = [picks[i] for i in assignments[assignments["event_idx"] == event_idx]["pick_idx"]]
        event = catalog.iloc[event_idx]

        first, last = min(pick.peak_time for pick in event_picks), max(pick.peak_time for pick in event_picks)

        sub = obspy.Stream()

        for station in np.unique([pick.trace_id for pick in event_picks]):
            sub.append(stream.select(station=station[3:-1], channel="HHZ")[0])

        sub = sub.slice(first - 5, last + 5)

        sub = sub.copy()
        sub.detrend()
        sub.filter("highpass", freq=2)

        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111)

        for i, trace in enumerate(sub):
            normed = trace.data - np.mean(trace.data)
            normed = normed / np.max(np.abs(normed))
            station_x, station_y = station_dict[trace.id[:-4]]
            y = np.sqrt((station_x - event["x(km)"]) ** 2 + (station_y - event["y(km)"]) ** 2 + event["z(km)"] ** 2)
            ax.plot(trace.times(), 10 * normed + y)
            
        for pick in event_picks:
            station_x, station_y = station_dict[pick.trace_id]
            y = np.sqrt((station_x - event["x(km)"]) ** 2 + (station_y - event["y(km)"]) ** 2 + event["z(km)"] ** 2)
            x = pick.peak_time - trace.stats.starttime
            if pick.phase == "P":
                ls = '-'
            else:
                ls = '--'
            ax.plot([x, x], [y - 10, y + 10], 'k', ls=ls)
            
        ax.set_ylim(0)
        ax.set_xlim(0, np.max(trace.times()))
        ax.set_ylabel("Hypocentral distance [km]")
        ax.set_xlabel("Time [s]")

        print("Event information")
        print(event)

        file_name = '{0}.{extention}'.format('./result/plots/associated_event', extention='png')
        fig.savefig(file_name, facecolor = 'w')


        



