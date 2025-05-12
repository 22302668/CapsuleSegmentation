import pandas as pd

def detect_stops(df, speed_thresh=2.0, min_duration_s=60):
    """
    Détecte les arrêts si la vitesse reste sous un seuil pendant une certaine durée.

    Args:
        df (pd.DataFrame): Données avec 'timestamp', 'speed_kmh'
        speed_thresh (float): seuil de vitesse (km/h)
        min_duration_s (float): durée minimale (secondes)

    Returns:
        pd.DataFrame: tableau des arrêts
    """
    df = df.copy()
    df['is_slow'] = df['speed_kmh'] < speed_thresh
    df['group'] = (df['is_slow'] != df['is_slow'].shift()).cumsum()

    stops = df[df['is_slow']].groupby('group').agg(
        start_time=('timestamp', 'min'),
        end_time=('timestamp', 'max'),
        duration_s=('time_diff_s', 'sum'),
        lat=('lat', 'mean'),
        lon=('lon', 'mean')
    ).reset_index(drop=True)

    return stops[stops['duration_s'] >= min_duration_s]
