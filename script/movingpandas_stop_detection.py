# stop_detection/movingpandas_stop_detection.py
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import movingpandas as mpd
from datetime import timedelta

def detect_stops_with_movingpandas(df, min_duration_minutes=8, max_diameter_meters=30):
    """
    Détecte les stops avec MovingPandas à partir d'un DataFrame contenant 'lat', 'lon', 'timestamp'.
    
    Args:
        df (pd.DataFrame): Données GPS avec colonnes 'lat', 'lon', 'timestamp'
        min_duration_minutes (int): Durée minimale d'un arrêt (en minutes)
        max_diameter_meters (float): Diamètre maximal de la zone de stop (en mètres)

    Returns:
        pd.DataFrame: Liste des stops détectés avec lat/lon, temps de début/fin, durée
    """
    print(f"Points d'entrée : {len(df)}")  # Affiche le nombre total de points GPS en entrée
    print(f"Timestamps min/max : {df['timestamp'].min()} / {df['timestamp'].max()}")  # Affiche les bornes temporelles du dataset

    # Conversion en GeoDataFrame avec géométrie
    df['geometry'] = df.apply(lambda row: Point(row['lon'], row['lat']), axis=1)  # Crée une géométrie Point pour chaque ligne à partir de lon/lat
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')  # Convertit le DataFrame en GeoDataFrame avec système de coordonnées WGS84

    # Créer une trajectoire
    trajectory = mpd.Trajectory(gdf, traj_id=1, t='timestamp')  # Crée une trajectoire MovingPandas avec un identifiant et les timestamps

    # Détecter les stops
    detector = mpd.TrajectoryStopDetector(trajectory)  # Instancie un détecteur d'arrêts pour cette trajectoire
    stop_points = detector.get_stop_points(
        min_duration=timedelta(minutes=min_duration_minutes),  # Durée minimale de stop
        max_diameter=max_diameter_meters  # Diamètre spatial maximal pour considérer un arrêt
    )

    print(f"{len(stop_points)} stops bruts détectés par MovingPandas")  # Affiche le nombre d’arrêts trouvés
    print("Colonnes retournées :", stop_points.columns.tolist())  # Affiche les noms des colonnes du résultat

    # Identifier dynamiquement les colonnes de début et fin
    if stop_points.empty:
        return pd.DataFrame(columns=['start_time', 'end_time', 'duration_s', 'lat', 'lon'])  # Retourne un DataFrame vide avec les bonnes colonnes si aucun stop n’est trouvé

    start_col = 't0' if 't0' in stop_points else 'start_time'  # Choix dynamique du nom de la colonne de début selon la version de MovingPandas
    end_col = 't1' if 't1' in stop_points else 'end_time'  # Idem pour la colonne de fin

    # Calcul durée + nettoyage
    stop_points['duration_s'] = (stop_points[end_col] - stop_points[start_col]).dt.total_seconds()  # Calcule la durée du stop en secondes
    stop_points = stop_points.rename(columns={
        start_col: 'start_time',  # Renomme la colonne de début
        end_col: 'end_time',      # Renomme la colonne de fin
        'geometry': 'stop_geom'   # Renomme la géométrie pour éviter conflit futur
    })

    stop_points['lat'] = stop_points['stop_geom'].y  # Extrait la latitude à partir de la géométrie
    stop_points['lon'] = stop_points['stop_geom'].x  # Extrait la longitude à partir de la géométrie

    print("\nAperçu des stops détectés :")  # Affiche un aperçu des résultats
    print(stop_points[['start_time', 'end_time', 'duration_s', 'lat', 'lon']].head())

    return stop_points[['start_time', 'end_time', 'duration_s', 'lat', 'lon']]  # Retourne le DataFrame avec les colonnes utiles