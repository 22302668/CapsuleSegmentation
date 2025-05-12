import pandas as pd
from geopy.distance import geodesic

def detect_stops(df, distance_thresh=50, duration_thresh=60):
    """
    Détecte les stops à partir d'une méthode de stay point (Zheng et al.).
    Un stop est défini par une séquence de points dans un rayon spatial pendant une durée minimale.
    
    Args:
        df (pd.DataFrame): Données avec 'lat', 'lon', 'timestamp'
        distance_thresh (float): distance max en mètres entre points
        duration_thresh (float): durée minimale en secondes

    Returns:
        pd.DataFrame: tableau des arrêts détectés
    """
    stops = []
    i = 0
    while i < len(df):
        j = i + 1
        while j < len(df):
            dist = geodesic((df.loc[i, 'lat'], df.loc[i, 'lon']),
                            (df.loc[j, 'lat'], df.loc[j, 'lon'])).meters
            if dist > distance_thresh:
                break
            j += 1

        if j - 1 > i:
            t_start = df.loc[i, 'timestamp']
            t_end = df.loc[j - 1, 'timestamp']
            duration = (t_end - t_start).total_seconds()
            if duration >= duration_thresh:
                lat_mean = df.loc[i:j - 1, 'lat'].mean()
                lon_mean = df.loc[i:j - 1, 'lon'].mean()
                stops.append({
                    'start_time': t_start,
                    'end_time': t_end,
                    'duration_s': duration,
                    'lat': lat_mean,
                    'lon': lon_mean
                })
                i = j
            else:
                i += 1
        else:
            i += 1
    return pd.DataFrame(stops)