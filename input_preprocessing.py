# import the required libraries
import pandas as pd
from h3 import h3
import webbrowser
import os
import folium
from folium import Map

class InputPreprocessing:
    df_original = pd.read_csv("data/input/VED_171101_week.csv")
    resolution = 10

    @classmethod
    def get_gps_logs(cls):
        df_raw_selected = cls.df_original[['DayNum', 'VehId', 'Trip', 'Timestamp(ms)', 'Latitude[deg]', 'Longitude[deg]']]
        df_raw_selected = df_raw_selected.rename(columns={'DayNum': 'day', 'VehId': 'vehicle_id',
                                                          'Trip': 'trip', 'Timestamp(ms)': 'ts', 'Latitude[deg]': 'lat',
                                                          'Longitude[deg]': 'lon'})
        df_gps_logs = df_raw_selected.sort_values(by=['day', 'vehicle_id', 'trip']).reset_index(drop=True)
        df_gps_logs = df_gps_logs[['day', 'vehicle_id', 'trip', 'lat', 'ts', 'lon']]
        df_gps_logs.to_csv('data/new_input/gps_logs.csv')
        return df_gps_logs


    @classmethod
    def preclustering_with_h3(cls, df_gps_logs):
        df_logs = df_gps_logs
        df_logs['h3'] = df_logs.apply(lambda row: h3.geo_to_h3(row['lat'], row['lon'], cls.resolution), axis=1)

        # clustering with h3 library
        h3_clusters = dict()
        cluster_id = 1
        cluster_id_col_value = []
        for index, row in df_logs.iterrows():
            key = row['h3']
            if key in h3_clusters:
                h3_clusters[key]["count"] += 1
            else:
                h3_clusters[key] = {
                    "cluster_id": cluster_id,
                    "count": 1,
                    "geom": h3.h3_to_geo_boundary(key)
                }
                cluster_id += 1
            cluster_id_col_value.append(h3_clusters[key]['cluster_id'])

        df_logs['cluster_id'] = cluster_id_col_value
        return h3_clusters, df_logs

    @classmethod
    def create_map(cls, clusters, df_gps_logs):
        # Create the map object
        map = Map(tiles="cartodbpositron",
                  attr='© <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors © <a href="http://cartodb.com/attributions#basemaps">CartoDB</a>')

        # Convert the clusters dictionary items to polygons and add them to the map
        for cluster in clusters.values():
            points = cluster['geom']
            # points = [p[::-1] for p in points]
            tooltip = "cluster_id = {0}, {1} trajectory sampling points".format(cluster['cluster_id'], cluster['count'])
            polygon = folium.vector_layers.Polygon(locations=points, tooltip=tooltip,
                                                   fill=True,
                                                   color='#ff0000',
                                                   fill_color='#ff0000',
                                                   fill_opacity=0.4, weight=3, opacity=0.4)
            polygon.add_to(map)

        # Determine the map bounding box
        max_lat = df_gps_logs.lat.max()
        min_lat = df_gps_logs.lat.min()
        max_lon = df_gps_logs.lon.max()
        min_lon = df_gps_logs.lon.min()

        # Fit the map to the bounds
        map.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

        return map

    @classmethod
    def show_map(cls, map, file_name):
        map.save(file_name)
        wb = webbrowser.open('file://' + os.path.realpath(file_name), new=2)

    @classmethod
    def visualize_clustering_results(cls, h3_clusters, df_gps_logs):
        relevant_clusters = {key: value for (key, value) in h3_clusters.items() if value['count'] >= 5}
        h3_map = cls.create_map(relevant_clusters, df_gps_logs)
        cls.show_map(h3_map, "map_h3_{0}.html".format(cls.resolution))

    @classmethod
    def extract_trajectory_points_with_h3_clustering(cls, df_logs):
        df_clustered_points = df_logs[['cluster_id', 'lat', 'lon', 'h3']].drop_duplicates()
        df_clustered_points.to_csv('data/new_input/clustered_points.csv')

    @classmethod
    def extract_trajectory_start_end_info(cls, df_logs):
        df_moves_selected = df_logs[['vehicle_id', 'day', 'ts', 'cluster_id']]
        df_moves_init = df_moves_selected[df_moves_selected['ts'] == 0]
        df_moves_init = df_moves_init.rename(columns={'ts': 'ts_ini', 'cluster_id': 'cluster_ini'})

        df_moves_end = df_moves_selected.copy()
        df_moves_end['ts_end'] = df_moves_end.groupby(by=['vehicle_id', 'day'])['ts'].transform("max")
        df_moves_end = df_moves_end[df_moves_end['ts'] == df_moves_end['ts_end']]
        df_moves_end = df_moves_end.drop(columns=['ts_end'])
        df_moves_end = df_moves_end.rename(columns={'ts': 'ts_end', 'cluster_id': 'cluster_end'})

        df_moves_full = pd.merge(df_moves_init, df_moves_end, how='inner', left_on=['vehicle_id', 'day'],
                                 right_on=['vehicle_id', 'day'])
        df_moves_full = df_moves_full[
            ['vehicle_id', 'day', 'ts_ini', 'ts_end', 'cluster_ini', 'cluster_end']].sort_values(by='vehicle_id')
        df_moves_full.to_csv('data/new_input/move_days.csv')

if __name__ == "__main__":
    df_gps_logs = InputPreprocessing.get_gps_logs()
    h3_clusters, df_preclustered_logs = InputPreprocessing.preclustering_with_h3(df_gps_logs)
    # print(h3_clusters)
    # InputPreprocessing.visualize_clustering_results(h3_clusters, df_preclustered_logs)
    InputPreprocessing.extract_trajectory_points_with_h3_clustering(df_preclustered_logs)
    InputPreprocessing.extract_trajectory_start_end_info(df_preclustered_logs)





