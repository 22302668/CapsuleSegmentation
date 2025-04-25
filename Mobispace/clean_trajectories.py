import pandas as pd
import geopandas as gpd
import movingpandas as mpd
from shapely.geometry import Point
from datetime import datetime, timedelta
import numpy as np
from geopy.distance import geodesic
import hvplot.pandas
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def load_data(data_folder):
    gps_file = os.path.join(data_folder, "GPS", "Participant9999961-gps.csv")
    measures_file = os.path.join(data_folder, "Measures", "Participant9999961-measures.csv")
    activities_file = os.path.join(data_folder, "Activities", "Participant9999961-activities.csv")

    gps_df = pd.read_csv(gps_file)
    measures_df = pd.read_csv(measures_file)
    activities_df = pd.read_csv(activities_file)

    for df, time_col in [(gps_df, "timestamp"), (measures_df, "time"), (activities_df, "time")]:
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        df.drop_duplicates(inplace=True)
        df.sort_values(by=[time_col], inplace=True)

    return gps_df, measures_df, activities_df

def create_geodataframe(df, lat_col='lat', lon_col='lon'):
    df['geometry'] = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf.set_crs(epsg=4326, inplace=True)
    return gdf

def segment_trajectories(gps_gdf, time_gap=timedelta(minutes=7), distance_gap=500):
    gps_gdf.set_index('timestamp', inplace=True)
    
    trajectory_ids = []
    current_trajectory = 0
    last_time = None
    last_location = None

    for index, row in gps_gdf.iterrows():
        current_location = (row["lat"], row["lon"])
        current_time = index if isinstance(index, pd.Timestamp) else pd.to_datetime(index)

        if last_time is not None and last_location is not None:
            time_diff = (current_time - last_time).total_seconds()
            distance_diff = geodesic(last_location, current_location).meters

            if pd.Timedelta(seconds=time_diff) > time_gap or distance_diff > distance_gap:
                current_trajectory += 1

        trajectory_ids.append(current_trajectory)
        last_location = current_location
        last_time = current_time

    gps_gdf['trajectory_id'] = trajectory_ids
    return mpd.TrajectoryCollection(gps_gdf, 'trajectory_id')

def clean_trajectories(trajectory_collection, min_points=5, alpha=2):
    cleaned_trajectories = []
    removed_points = {}

    for trajectory in trajectory_collection:
        cleaner = mpd.OutlierCleaner(trajectory)
        original_size = len(trajectory.df)
        
        cleaned_traj = cleaner.clean(alpha=alpha)
        cleaned_traj.add_speed(overwrite=True)
        
        cleaned_size = len(cleaned_traj.df)
        removed_points[trajectory.id] = original_size - cleaned_size

        if cleaned_size > min_points:
            cleaned_trajectories.append(cleaned_traj)

    return mpd.TrajectoryCollection(cleaned_trajectories), removed_points

def visualize_trajectories(gps_gdf, cleaned_gdf):
    map_original = gps_gdf.hvplot(
        geo=True, tiles='OSM', c='trajectory_id',
        line_width=1, width=800, height=600, alpha=0.4, legend=True, label="Original"
    )

    map_cleaned = cleaned_gdf.hvplot(
        geo=True, tiles='OSM', c='trajectory_id',
        line_width=2, width=800, height=600, alpha=1, legend=True, label="Cleaned"
    )

    return map_original * map_cleaned

def save_cleaned_trajectories(cleaned_gdf, data_folder):
    output_path = os.path.join(data_folder, "Cleaned_Trajectories.gpkg")
    cleaned_gdf.to_file(output_path, driver='GPKG', layer='trajectories')
    return output_path

def visualize_single_trajectory(trajectory_collection, cleaned_trajectories):
    traj_before = trajectory_collection.trajectories[0]
    traj_after = cleaned_trajectories[0]

    hvplot_defaults = {
        "tiles": "CartoLight",
        "frame_height": 400,
        "frame_width": 500,
        "cmap": "Viridis",
        "colorbar": True
    }

    return traj_before.hvplot(
        label="Avant Nettoyage", color="red", line_width=4, **hvplot_defaults
    ) * traj_after.hvplot(
        label="Après Nettoyage", c="speed", line_width=4, **hvplot_defaults
    )

def main():
    data_folder = "participant-data-semain43"
    
    gps_df, measures_df, activities_df = load_data(data_folder)
    gps_gdf = create_geodataframe(gps_df)
    trajectory_collection = segment_trajectories(gps_gdf)
    
    print(f"Nombre total de trajectoires détectées : {len(trajectory_collection)}")
    
    cleaned_collection, removed_points = clean_trajectories(trajectory_collection)
    cleaned_gdf = pd.concat([t.to_point_gdf() for t in cleaned_collection.trajectories])
    
    print(f"Nombre total de trajectoires après nettoyage : {len(cleaned_collection)}")
    
    removed_df = pd.DataFrame(list(removed_points.items()), columns=['Trajectory ID', 'Points Removed'])
    removed_df["Percentage Removed"] = (removed_df["Points Removed"] / removed_df["Points Removed"].sum()) * 100
    print("\nDistribution des points supprimés par trajectoire :")
    print(removed_df.describe())
    
    output_path = save_cleaned_trajectories(cleaned_gdf, data_folder)
    print(f"\nTrajectoires nettoyées sauvegardées sous '{output_path}'")

if __name__ == "__main__":
    main() 