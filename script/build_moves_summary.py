import pandas as pd
from geopy.distance import geodesic

def build_moves_summary(ds2: pd.DataFrame) -> pd.DataFrame:
    """
    Génère un tableau résumé des moves à partir des points GPS hors stops.

    Args:
        ds2 (pd.DataFrame): doit contenir ['timestamp', 'lat', 'lon']

    Returns:
        pd.DataFrame: tableau résumé avec les colonnes :
            ['start_time', 'end_time', 'duration_s',
             'lat_start', 'lon_start', 'lat_end', 'lon_end', 'dist_m']
    """
    # Tri chronologique
    ds2 = ds2.sort_values('timestamp').reset_index(drop=True)

    # On garde les couples consécutifs (ligne i -> ligne i+1)
    data = []
    for i in range(len(ds2) - 1):
        row_start = ds2.iloc[i]
        row_end = ds2.iloc[i + 1]

        start_time = row_start['timestamp']
        end_time   = row_end['timestamp']
        duration_s = (end_time - start_time).total_seconds()

        lat_start, lon_start = row_start['lat'], row_start['lon']
        lat_end, lon_end     = row_end['lat'], row_end['lon']
        dist_m = geodesic((lat_start, lon_start), (lat_end, lon_end)).meters

        data.append({
            'start_time': start_time,
            'end_time': end_time,
            'duration_s': duration_s,
            'lat_start': lat_start,
            'lon_start': lon_start,
            'lat_end': lat_end,
            'lon_end': lon_end,
            'dist_m': dist_m
        })

    return pd.DataFrame(data)
