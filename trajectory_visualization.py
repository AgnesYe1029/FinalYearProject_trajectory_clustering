import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from trajectory_clustering import TrajectoryClustering
import matplotlib as mpl

class TrajectoryVisualization:
    colors = ['darkgreen', 'red', 'purple', 'darkblue', 'orange', 'cyan', 'gray',
                     'darkred', 'darkgreen', 'darkorange', 'pink', 'greenyellow', 'skyblue', 'black',
                     'forestgreen', 'deeppink', 'violet', 'lightblue', 'steelblue', 'yellowgreen',
                     'seagreen', 'blueviolet', 'forestgreen', 'yellow', 'lightgreen']
    markers = ['H', 's', 'd', 'p', '*', 'X', 'P', 'D', 'h', '8', 'v', '^', '<', '>', 'o']

    def __init__(self, map_loc='map/', dpi=150, subplots=(2, 3)):
        roads = gpd.GeoDataFrame.from_file(map_loc + 'ann_map.shp')
        rows, cols = subplots
        self.fig, self.axis = plt.subplots(rows, cols, dpi=dpi, figsize=(6.4, 3.6))
        plt.subplots_adjust(top=0.98, bottom=0.02, left=0.05, right=0.95, hspace=0.02)
        if rows * cols > 1:
            self.axes = self.axes.flatten()
        else:
            self.axes = [self.axes]
        for axis in self.axes:
            axis.set_aspect('equal')
            axis.grid()
            roads.plot(ax=axis, color='gray', edgecolor='gray', linewidth=.1)
            axis.set_xlim(103.6, 104.04)
            axis.set_ylim(1.22, 1.475)
            axis.axes.xaxis.set_ticklabels([])
            axis.axes.yaxis.set_ticklabels([])
        if mpl.get_backend() == 'Qt5Agg':
            fig_manager = plt.get_current_fig_manager()
            fig_manager.window.showMaximized()


    @staticmethod
    def plot_show():
        plt.show()

    def zoom_fit(self, all_logs, margin=0.002):
        latitude_longtitude_all = np.concatenate([trajectory for traj_group in all_logs for trajectory in traj_group], axis=0)
        longtitude_delta = 0.03
        min_lat, max_lat = latitude_longtitude_all[:, 0].min(), latitude_longtitude_all[:, 0].max()
        min_lon, max_lon = latitude_longtitude_all[:, 1].min(), latitude_longtitude_all[:, 1].max()
        for axis in self.axes:
            axis.set_xlim(min_lon - margin, max_lon + margin)
            axis.set_ylim(min_lat - margin, max_lat + margin)

    def plot_trajectories(self, trajectories, clusters):
        for i, trip_cluster in enumerate(trajectories):
            axis = self.axis[i]
            start, dest = clusters[i][0], clusters[i][1]
            for trip in trip_cluster:
                axis.plot(trip[:, 1], trip[:, 0], color=self.select_colors[i % 25])
                starting_location = TrajectoryClustering.load_cluster(start)
                dest_location = TrajectoryClustering.load_cluster(dest)
                axis.text(starting_location[1], starting_location[0], "start", fontsize=8, color='black',
                          horizontalalignment='left', verticalalignment='bottom')
                axis.text(dest_location[1], dest_location[0], "end", fontsize=8, color='black',
                          horizontalalignment='left', verticalalignment='bottom')
                axis.scatter([starting_location[1], dest_location[1]], [starting_location[0], dest_location[0]])


