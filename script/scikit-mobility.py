# scikit_mobility_stop_detection.py

import pandas as pd
from skmob.preprocessing import detection

def detect_stops_with_skmob(
    df: pd.DataFrame,
    epsilon_m: float = 30,
    min_time_s: int = 6 * 60
) -> pd.DataFrame:
    """
    Détecte les stops avec scikit‐mobility.
    
    Args:
        df (pd.DataFrame): doit contenir ['lat','lon','timestamp'] tz‐aware.
        epsilon_m (float): rayon max (m) pour grouper les points en stop.
        min_time_s (int): durée min (s) pour qu’un arrêt soit validé.
    
    Returns:
        pd.DataFrame: colonnes ['start_time','end_time','duration_s','lat','lon'].
    """
    # 1) Préparer data pour scikit-mobility
    skdf = df[['lat','lon','timestamp']].rename(columns={'timestamp':'datetime'}).copy()
    skdf['datetime'] = pd.to_datetime(skdf['datetime'])
    
    # 2) Appel à scikit-mobility
    stops = detection.stops(
        skdf,
        epsilon=epsilon_m,
        min_time=min_time_s
    )
    # stops a ['lat','lon','t_in','t_out','npts']
    
    # 3) Renommage / calcul durée
    stops = stops.rename(columns={'t_in':'start_time','t_out':'end_time'})
    stops['duration_s'] = (stops['end_time'] - stops['start_time']).dt.total_seconds()
    
    return stops[['start_time','end_time','duration_s','lat','lon']]
