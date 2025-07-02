import pandas as pd
import numpy as np
from geopy.distance import geodesic
import os
from sqlalchemy import text
import geopandas as gpd

def load_data_and_prepare(engine, participant_id):
    """
    Charge les données GPS depuis PostgreSQL, calcule la distance, vitesse, vitesse lissée, etc.
    """

    with engine.connect() as conn:
        df = pd.read_sql_query(
            text(f"SELECT * FROM gps_mesures WHERE participant_virtual_id = '{participant_id}'"),
            con=conn
        )

    # Preprocessing
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('Europe/Paris')

    df = df.sort_values(by='timestamp').reset_index(drop=True)

    df['time_diff_s'] = df['timestamp'].diff().dt.total_seconds()

    distances = [None]
    for i in range(1, len(df)):
        p1 = (df.loc[i-1, 'lat'], df.loc[i-1, 'lon'])
        p2 = (df.loc[i, 'lat'], df.loc[i, 'lon'])
        distances.append(geodesic(p1, p2).meters)
    df['dist_m'] = distances

    df['speed_kmh'] = df['dist_m'] / df['time_diff_s'] * 3.6
    df['speed_kmh'] = df['speed_kmh'].replace([np.inf, -np.inf], np.nan)
    #df = df[df['speed_kmh'] <= 150]
    df['speed_kmh_smooth'] = df['speed_kmh'].rolling(window=5, min_periods=1, center=True).mean()

    return df

def segment_by_data_weeks(df):
    """
    A partir d'un DataFrame contenant au moins la colonne 'timestamp' (datetime),
    détecte automatiquement les plages de dates consécutives où il y a au moins
    un point GPS. Renvoie une liste de tuples (date_debut, date_fin).
    """
    unique_dates = sorted(df['timestamp'].dt.date.unique())
    if not unique_dates:
        return []

    segments = []
    start = unique_dates[0]
    prev = unique_dates[0]
    for current in unique_dates[1:]:
        if (current - prev).days > 1:
            segments.append((start, prev))
            start = current
        prev = current
    segments.append((start, prev))
    return segments