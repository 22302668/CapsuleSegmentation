import pandas as pd
import geopandas as gpd
import movingpandas as mpd
from movingpandas import StopSplitter
from datetime import timedelta
import numpy as np
import os
import warnings

#ignorer les avertissements inutiles
warnings.filterwarnings("ignore", category=UserWarning)

#detection pour les stops dans une collection de trajectoires
def detect_stops(trajectory_collection, max_diameter=50, min_duration=timedelta(minutes=7)):
    """
    - max_diameter: diametre maximal d'un arret en metres, si une personne reste dans un rayon de 50m ça serait considéré comme un arret
    - min_duration: duree minimale d'un arret, 7 minutes souvent utilisé, Si une personne reste immobile plus longtemps que ce seuil c'est donc un arrêt
    """
    detector = mpd.TrajectoryStopDetector(trajectory_collection)
    stop_segments = detector.get_stop_segments(
        min_duration=min_duration,
        max_diameter=max_diameter
    )
    return stop_segments

#segmentation des trajectoires en stops et moves
def split_trajectories(trajectory_collection, max_diameter=100, min_duration=timedelta(minutes=7)):
    """
    - max_diameter: diametre maximal pour identifier un arret (par defaut 100m).
    - min_duration: duree minimale pour definir un arret (par defaut 7 minutes).
    """
    stop_splitter = StopSplitter(trajectory_collection)
    segmented_trajectories = stop_splitter.split( #segmenter la trajectoire en plusieurs morceaux avec split 
        max_diameter=max_diameter,
        min_duration=min_duration
    )
    return segmented_trajectories

#separer les stops et moves
def separate_stops_moves(segmented_trajectories, stop_threshold=5):
    stops = []
    moves = []
    """
    - stop_threshold: duree minimale pour considerer un trajet comme un mouvement
    """
    for traj in segmented_trajectories.trajectories:
        duration_minutes = (traj.get_end_time() - traj.get_start_time()).total_seconds() / 60  #le temps entre le debut et la fin d'une trajectoire et ça convertit cette duree en minutes
        if duration_minutes < stop_threshold:  #si le segment dure moins de 5 minutes c'est un arret
            stops.append(traj)
        else:
            moves.append(traj)

    return stops, moves

#calculer la durre des stops et des moves en minutes
def calculate_durations(stops, moves):
    stop_durations = [traj.get_duration().total_seconds() / 60 for traj in stops]
    move_durations = [traj.get_duration().total_seconds() / 60 for traj in moves]
    return stop_durations, move_durations

#convertir les stops et moves en GeoDataFrame
def convert_to_geodataframe(stops, moves):
    moves_gdf = mpd.TrajectoryCollection(moves).to_line_gdf()
    stops_gdf = mpd.TrajectoryCollection(stops).to_point_gdf()
    return moves_gdf, stops_gdf

#assigner les activités aux stops et moves en fonction de leur timestamp.
def assign_activities(stops_gdf, moves_gdf, activities_df):
    """
    - activities_df: dataFrame contenant les timestamps et des activités correspondantes aux stops et moves
    """
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

#sauvegarder les shapefiles
def save_shapefiles(stops_gdf, moves_gdf, data_folder):
    #verification de géo si est bien définie pour les moves
    if moves_gdf.crs is None:
        moves_gdf.set_crs(epsg=4326, inplace=True)  #definit le systeme de coordonnées
    
    #garde seulement les colonnes necessaires pour sauvegarder le shapefile des deplacements
    moves_gdf = moves_gdf[['geometry', 'trajectory_id']].copy()
    moves_shapefile_path = os.path.join(data_folder, "moves.shp")
    moves_gdf.to_file(moves_shapefile_path)  #sauvegarde le fichier shapefile pour les deplacements

    #si les stops ne sont pas en format GeoDataFrame les convertir
    if not isinstance(stops_gdf, gpd.GeoDataFrame):
        stops_gdf = gpd.GeoDataFrame(stops_gdf, geometry='geometry')
    
    #verification de géo si est bien définie pour les stops
    if stops_gdf.crs is None:
        stops_gdf.set_crs(epsg=4326, inplace=True)

    stops_gdf = stops_gdf.reset_index()  #reinitialise l'index pour éviter les erreurs lors de l'export
    stops_gdf = stops_gdf[['geometry', 'activity']].copy()
    stops_gdf = stops_gdf.rename(columns={'activity': 'act'})  #renomme la colonne pour simplifier le nommage
    stops_gdf['act'] = stops_gdf['act'].fillna('None')  #remplace les valeurs manquantes par None

    stops_shapefile_path = os.path.join(data_folder, "centroids.shp")
    stops_gdf.to_file(stops_shapefile_path)  #sauvegarde le fichier shapefile pour les stops

    return moves_shapefile_path, stops_shapefile_path

#visualiser les stops et moves sur une carte
def visualize_stops_moves(stops_gdf, moves_gdf):
    hvplot_defaults = {
        "tiles": "OSM",  #utiliser la carte OpenStreetMap
        "frame_height": 500,
        "frame_width": 800,
        "cmap": "Viridis",
        "colorbar": True
    }

    #visualiser les stops en rouge
    stops_map = stops_gdf.hvplot(
        geo=True, size=100, alpha=0.6, color="red",
        **hvplot_defaults, label="Stops"
    )

    #visualiser les moves en bleu
    moves_map = moves_gdf.hvplot(
        geo=True, line_width=2, alpha=1, color="blue",
        **hvplot_defaults, label="Moves"
    )

    return stops_map * moves_map  #combine les deux cartes pour une visualisation complete

def main():
    data_folder = "participant-data-semain43"
    
    #charger les données nettoyées
    cleaned_trajectories = gpd.read_file(os.path.join(data_folder, "Cleaned_Trajectories.gpkg"), layer='trajectories')
    activities_df = pd.read_csv(os.path.join(data_folder, "Activities", "Participant9999961-activities.csv")) #pourquoi recharger les activités ici?
    
    # Créer la collection de trajectoires
    tc = mpd.TrajectoryCollection(cleaned_trajectories, 'trajectory_id', t='timestamp')
    
    # Sélectionner un sous-ensemble de trajectoires
    subset_trajectories = tc.trajectories[:50] #prendre les 50 premieres trajectoires
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