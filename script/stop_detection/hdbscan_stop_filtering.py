import pandas as pd

def detect_stops(df, speed_thresh=1.0, min_duration_s=60):
    """
    Détecte les arrêts uniquement dans le bruit (cluster -1) si vitesse < seuil et durée > seuil.

    Args:
        df (pd.DataFrame): DataFrame contenant au moins les colonnes 'timestamp', 'lat', 'lon', 'speed_kmh', 'cluster_behavior'.
        speed_thresh (float): seuil de vitesse (km/h) pour considérer un point lent.
        min_duration_s (int): durée minimale d'un arrêt (en secondes).

    Returns:
        pd.DataFrame: tableau des arrêts (start_time, end_time, duration_s, lat, lon)
    """
    if 'cluster_behavior' not in df.columns:
        raise ValueError("Colonne 'cluster_behavior' manquante. Exécutez le clustering HDBSCAN d'abord.")

    df_noise = df[(df['cluster_behavior'] == -1) & (df['speed_kmh'] < speed_thresh)].copy()
    if df_noise.empty:
        return pd.DataFrame(columns=['start_time', 'end_time', 'duration_s', 'lat', 'lon'])

    df_noise['gap'] = df_noise['timestamp'].diff().dt.total_seconds().fillna(0)
    df_noise['group'] = (df_noise['gap'] > min_duration_s).cumsum()

    stops = df_noise.groupby('group').agg(
        start_time=('timestamp', 'min'),
        end_time=('timestamp', 'max'),
        duration_s=('time_diff_s', 'sum'),
        lat=('lat', 'mean'),
        lon=('lon', 'mean')
    ).reset_index(drop=True)

    return stops[stops['duration_s'] >= min_duration_s]