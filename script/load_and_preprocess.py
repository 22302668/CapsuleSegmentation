import pandas as pd
import numpy as np
from geopy.distance import geodesic
import os
from sqlalchemy import text
import geopandas as gpd

def load_data_and_prepare(engine):
    """
    Charge les données GPS depuis PostgreSQL, calcule la distance, vitesse, vitesse lissée, etc.
    """
    # current_dir = os.getcwd()  #chemin vers le répertoire contenant le script
    # data_folder = os.path.abspath(os.path.join(current_dir, "..", "participant-data-semain43"))  #chemin vers le repertoire contenant le script
    # data_path = os.path.join(data_folder, "Cleaned_Trajectories.gpkg")  #eonstruction du chemin absolu vers le dossier contenant les données

    # print("Chargement des trajectoires...")
    # df = gpd.read_file(data_path, layer='trajectories')  #lecture du fichier GeoPackage avec GeoPandas, couche 'trajectories'
    # print("Colonnes disponibles :", df.columns)

    # path = r"C:\Users\22302668\Desktop\CapsuleV2\participant-data-semain43\GPS\Participant9999965-gps.csv"
    # df = pd.read_csv(path)
    # df.columns = df.columns.str.strip().str.replace('"', '')  # ← nettoie les noms de colonnes
    with engine.connect() as conn:
        df = pd.read_sql_query(text("SELECT * FROM clean_gps WHERE participant_virtual_id = '9999965'"), con=conn)

    # Preprocessing
    df['timestamp'] = pd.to_datetime(df['time'], utc=True).dt.tz_convert('Europe/Paris')
    # df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('Europe/Paris')

    df = df.drop(columns=['time'])
    # df = df.sort_values(by='timestamp').reset_index(drop=True)

    df['time_diff_s'] = df['timestamp'].diff().dt.total_seconds()

    distances = [None]
    for i in range(1, len(df)):
        p1 = (df.loc[i-1, 'lat'], df.loc[i-1, 'lon'])
        p2 = (df.loc[i, 'lat'], df.loc[i, 'lon'])
        distances.append(geodesic(p1, p2).meters)
    df['dist_m'] = distances

    df['speed_kmh'] = df['dist_m'] / df['time_diff_s'] * 3.6
    df['speed_kmh'] = df['speed_kmh'].replace([np.inf, -np.inf], np.nan)
    df = df[df['speed_kmh'] <= 150]
    df['speed_kmh_smooth'] = df['speed_kmh'].rolling(window=5, min_periods=1, center=True).mean()

    return df
