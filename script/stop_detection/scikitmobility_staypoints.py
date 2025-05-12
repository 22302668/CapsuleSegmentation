from skmob import TrajDataFrame
from skmob.preprocessing import detection
import pandas as pd

def detect_stops(df, stop_radius_factor=0.2, minutes_for_a_stop=5):
    """
    Détection des stops avec scikit-mobility (stay_locations).
    Nécessite les colonnes 'lat', 'lon', 'timestamp' et 'uid'.

    Args:
        df (pd.DataFrame): Données GPS.
        stop_radius_factor (float): tolérance spatiale.
        minutes_for_a_stop (int): durée minimale d’un arrêt (minutes).

    Returns:
        pd.DataFrame: DataFrame contenant lat, lon, start_time, end_time, duration_s
    """
    if 'uid' not in df.columns:
        df['uid'] = 1  # valeur unique par défaut

    df = df.rename(columns={"timestamp": "datetime", "lon": "lng"})
    tdf = TrajDataFrame(df, timestamp=True)
    stop_df = detection.stay_locations(
        tdf, stop_radius_factor=stop_radius_factor, minutes_for_a_stop=minutes_for_a_stop
    )

    if stop_df.empty or 't0' not in stop_df.columns or 't1' not in stop_df.columns:
        return pd.DataFrame(columns=['start_time', 'end_time', 'duration_s', 'lat', 'lon'])

    stop_df['duration_s'] = (stop_df['t1'] - stop_df['t0']).dt.total_seconds()

    return stop_df.rename(columns={'lat': 'lat', 'lng': 'lon', 't0': 'start_time', 't1': 'end_time'})[
        ['start_time', 'end_time', 'duration_s', 'lat', 'lon']
    ]
