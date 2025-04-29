import pandas as pd
import numpy as np
from geopy.distance import geodesic

def load_data_and_prepare(engine):
    """
    Charge les données GPS depuis PostgreSQL, calcule la distance, vitesse, vitesse lissée, etc.
    """
    from sqlalchemy import text

    with engine.connect() as conn:
        df = pd.read_sql_query(text("SELECT * FROM clean_gps WHERE participant_virtual_id = '9999932'"), con=conn)

    # Preprocessing
    df['timestamp'] = pd.to_datetime(df['time'], utc=True).dt.tz_convert('Europe/Paris')
    df = df.drop(columns=['time'])
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
    df = df[df['speed_kmh'] <= 150]
    df['speed_kmh_smooth'] = df['speed_kmh'].rolling(window=5, min_periods=1, center=True).mean()

    return df
