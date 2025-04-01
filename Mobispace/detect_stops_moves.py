import pandas as pd
import geopandas as gpd
import movingpandas as mpd
from movingpandas import StopSplitter
from datetime import timedelta
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def detect_stops(trajectory_collection, max_diameter=50, min_duration=timedelta(minutes=7)):
    detector = mpd.TrajectoryStopDetector(trajectory_collection)
    stop_segments = detector.get_stop_segments(
        min_duration=min_duration,
        max_diameter=max_diameter
    )
    return stop_segments

def split_trajectories(trajectory_collection, max_diameter=100, min_duration=timedelta(minutes=7)):
    stop_splitter = StopSplitter(trajectory_collection)
    segmented_trajectories = stop_splitter.split(
        max_diameter=max_diameter,
        min_duration=min_duration
    )
    return segmented_trajectories

def separate_stops_moves(segmented_trajectories, stop_threshold=5):
    stops = []
    moves = []

    for traj in segmented_trajectories.trajectories:
        duration_minutes = (traj.get_end_time() - traj.get_start_time()).total_seconds() / 60
        if duration_minutes < stop_threshold:
            stops.append(traj)
        else:
            moves.append(traj)

    return stops, moves

def calculate_durations(stops, moves):
    stop_durations = [traj.get_duration().total_seconds() / 60 for traj in stops]
    move_durations = [traj.get_duration().total_seconds() / 60 for traj in moves]
    return stop_durations, move_durations

def convert_to_geodataframe(stops, moves):
    moves_gdf = mpd.TrajectoryCollection(moves).to_line_gdf()
    stops_gdf = mpd.TrajectoryCollection(stops).to_point_gdf()
    return moves_gdf, stops_gdf

def assign_activities(stops_gdf, moves_gdf, activities_df):
    def assign_activity(timestamp):
        previous_activity = None
        for _, row in activities_df.iterrows():
            if row["time"] <= timestamp:
                previous_activity = row["activity"]
            else:
                break
        return previous_activity

    stops_gdf.index = pd.to_datetime(stops_gdf.index, utc=True)
    moves_gdf["t"] = pd.to_datetime(moves_gdf["t"], utc=True)
    activities_df["time"] = pd.to_datetime(activities_df["time"], utc=True)

    stops_gdf["activity"] = stops_gdf.index.map(assign_activity)
    moves_gdf["activity"] = moves_gdf["t"].map(assign_activity)

    return stops_gdf, moves_gdf

def save_shapefiles(stops_gdf, moves_gdf, data_folder):
    if moves_gdf.crs is None:
        moves_gdf.set_crs(epsg=4326, inplace=True)
    
    moves_gdf = moves_gdf[['geometry', 'id']].copy()
    moves_shapefile_path = os.path.join(data_folder, "moves.shp")
    moves_gdf.to_file(moves_shapefile_path)

    if not isinstance(stops_gdf, gpd.GeoDataFrame):
        stops_gdf = gpd.GeoDataFrame(stops_gdf, geometry='geometry')
    
    if stops_gdf.crs is None:
        stops_gdf.set_crs(epsg=4326, inplace=True)

    stops_gdf = stops_gdf.reset_index()
    stops_gdf = stops_gdf[['geometry', 'activity']].copy()
    stops_gdf = stops_gdf.rename(columns={'activity': 'act'})
    stops_gdf['act'] = stops_gdf['act'].fillna('None')

    stops_shapefile_path = os.path.join(data_folder, "centroids.shp")
    stops_gdf.to_file(stops_shapefile_path)

    return moves_shapefile_path, stops_shapefile_path

def visualize_stops_moves(stops_gdf, moves_gdf):
    hvplot_defaults = {
        "tiles": "OSM",
        "frame_height": 500,
        "frame_width": 800,
        "cmap": "Viridis",
        "colorbar": True
    }

    stops_map = stops_gdf.hvplot(
        geo=True, size=100, alpha=0.6, color="red",
        **hvplot_defaults, label="Stops"
    )

    moves_map = moves_gdf.hvplot(
        geo=True, line_width=2, alpha=1, color="blue",
        **hvplot_defaults, label="Moves"
    )

    return stops_map * moves_map

def main():
    data_folder = "participant-data-semain43"
    
    # Charger les données nettoyées
    cleaned_trajectories = pd.read_csv(os.path.join(data_folder, "Cleaned_Trajectories.csv"))
    activities_df = pd.read_csv(os.path.join(data_folder, "Activities", "Participant9999961-activities.csv"))
    
    # Créer la collection de trajectoires
    tc = mpd.TrajectoryCollection(cleaned_trajectories, 'trajectory_id', t='timestamp')
    
    # Sélectionner un sous-ensemble de trajectoires
    subset_trajectories = tc.trajectories[:50]
    tc_subset = mpd.TrajectoryCollection(subset_trajectories, 'trajectory_id', t='timestamp')
    
    # Détecter les stops et moves
    segmented_trajectories = split_trajectories(tc_subset)
    stops, moves = separate_stops_moves(segmented_trajectories)
    
    # Calculer les durées
    stop_durations, move_durations = calculate_durations(stops, moves)
    
    # Convertir en GeoDataFrame
    moves_gdf, stops_gdf = convert_to_geodataframe(stops, moves)
    
    # Assigner les activités
    stops_gdf, moves_gdf = assign_activities(stops_gdf, moves_gdf, activities_df)
    
    # Sauvegarder les shapefiles
    moves_path, stops_path = save_shapefiles(stops_gdf, moves_gdf, data_folder)
    
    print(f"Nombre de stops détectés : {len(stops)}")
    print(f"Nombre de moves détectés : {len(moves)}")
    print(f"Durée moyenne des stops : {np.mean(stop_durations):.2f} minutes" if len(stops) > 0 else "Aucun stop détecté")
    print(f"Durée moyenne des moves : {np.mean(move_durations):.2f} minutes" if len(moves) > 0 else "Aucun move détecté")
    print(f"\nFichiers sauvegardés:\n- Moves: {moves_path}\n- Centroids: {stops_path}")

if __name__ == "__main__":
    main() 