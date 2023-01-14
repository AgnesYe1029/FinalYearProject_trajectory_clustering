import numpy as np
import pandas as pd

from rdp_reduction import rdp_reduction, indexed_rdp_reduction
import utm
from sklearn.cluster import DBSCAN
import time
import similaritymeasures

class TrajectoryClustering:
    df_move_days = pd.read_csv("data/input/move_days.csv")
    df_clusters = pd.read_csv("data/input/clusters.csv")
    df_logs = pd.read_csv("data/input/gps_logs.csv")

    @classmethod
    def get_trajectories(cls, start_cluster, end_cluster):
        df_move = cls.df_move_days
        df_log = cls.df_logs
        f1 = df_move["cluster_ini"] == start_cluster
        f2 = df_move["cluster_end"] == end_cluster
        moves = cls.df_move_days.loc[f1 & f2]
        trajectories = []
        for index, move in moves.iterrows():
            day_number = move["day"]
            vehicle = move["vehicle_id"]
            condition1 = df_log["day"] == day_number
            condition2 = df_log["vehicle_id"] == vehicle
            condition3 = df_log["lat"] != df_log["lat"].shift(1)
            condition4 = df_log["lon"] != df_log["lon"].shift(1)
            cols = ["lat", "lon"]
            df_assorted_log = df_log.loc[condition1 & condition2 & condition3 & condition4, cols]
            if not df_assorted_log.empty:
                trajectories.append(df_assorted_log[cols].values)
        return trajectories

    @staticmethod
    def latitude_longtitude_coord_converstion(trajectories):
        trajectories_coordinates = []
        for index, traj in enumerate(trajectories):
            coords = np.array([list(utm.from_latlon(ll[0], ll[1])[:2]) for ll in traj])
            trajectories_coordinates.append(coords)
        return trajectories_coordinates

    @classmethod
    def compute_distance_matrix(cls, trajectories, method = "Frechet"):

        traj_len = len(trajectories)
        distance_matrix = np.zeros((traj_len, traj_len))
        for i in range(traj_len-1):
            m = trajectories[i]
            for j in range(i + 1, traj_len):
                n = trajectories[j]
                if method == "Frechet":
                    distance_matrix[i, j] = similaritymeasures.frechet_dist(m, n)
                else:
                    distance_matrix[i, j] = similaritymeasures.area_between_two_curves(m, n)
                distance_matrix[j, i] = distance_matrix[i, j]
        return distance_matrix

    @classmethod
    def rdp_polyline_reduction(cls, line, epsilon = 10, return_indices = False):
        point_set = line.tolist()
        if return_indices:
            indices = list(range(len(point_set)))
            pts, indices = indexed_rdp_reduction(point_set, indices, epsilon)
            return np.array(pts), np.array(indices)
        else:
            pts = rdp_reduction(point_set, epsilon)
            return np.array(pts)

    @classmethod
    def dbscan_clustering(cls,distance_matrix, epsilon = 100):
        clusters = DBSCAN(eps=epsilon, min_samples=1, metric="precomputed")
        clusters.fit(distance_matrix)
        return clusters.labels_

    @classmethod
    def load_cluster(cls, cluster_id):
        df = cls.df_clusters
        locations = df.loc[df["cluster_id"] == cluster_id, ["lat", "lon"]].values
        return locations[0]

if __name__ == "__main__":
    clusters = [[1, 184], [144, 11], [1, 173], [11, 22], [29, 184], [29, 185]]
    cluster_start, cluster_end = clusters[1]

    trajectories = TrajectoryClustering.get_trajectories(cluster_start, cluster_end)
    print("number of trajectories: {}".format(len(trajectories)))
    trajectories_xy = TrajectoryClustering.latitude_longtitude_coord_converstion(trajectories)
    t0 = time.time()
    dist_mat = TrajectoryClustering.compute_distance_matrix(trajectories_xy)
    t1 = time.time()
    print("distance matrix without rdp completes in {} seconds".format(t1 - t0))
    t0 = time.time()
    trajectories_reduced = [TrajectoryClustering.rdp_polyline_reduction(p) for p in trajectories_xy]
    dist_mat_reduced = TrajectoryClustering.compute_distance_matrix(trajectories_reduced)
    t1 = time.time()
    print("distance matrix with rdp completes in {} seconds.".format(t1 - t0))
    data = [[len(t), len(trajectories_reduced[i])] for i, t in enumerate(trajectories)]
    df = pd.DataFrame(data=data, columns=["original data points", "after RDP"])
    df.index.name = "traj #"
    df.to_csv("data/output/RDP_data_points.csv")



