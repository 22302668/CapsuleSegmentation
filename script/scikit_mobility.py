import pandas as pd
from skmob import TrajDataFrame
from skmob.preprocessing.detection import stay_locations

def detect_stops_with_skmob(
    df: pd.DataFrame,
    epsilon_m: float = 75,
    min_time_s: int = 15 * 60
) -> pd.DataFrame:
    """
    Détecte les stops avec scikit-mobility (fonction stay_locations).

    Args:
        df (pd.DataFrame): DataFrame GPS avec colonnes ['lat','lon','timestamp'].
        epsilon_m (float): rayon max (m) pour grouper les points en stop.
        min_time_s (int): durée min (s) pour qu'un arrêt soit validé.

    Returns:
        pd.DataFrame: colonnes ['start_time','end_time','duration_s','lat','lon'].
    """
    # 1) Préparer le TrajDataFrame
    skdf = df[['lat', 'lon', 'timestamp']].rename(columns={'timestamp': 'datetime'}).copy()
    skdf['uid'] = 1  # un même utilisateur pour tout le segment
    tdf = TrajDataFrame(
        skdf,
        latitude='lat',
        longitude='lon',
        datetime='datetime',
        user_id='uid'
    )

    # 2) Appel à stay_locations
    spatial_radius_km   = epsilon_m / 1000.0      # convertir mètres → kilomètres
    minutes_for_a_stop  = min_time_s / 60.0       # convertir secondes → minutes
    stays = stay_locations(
        tdf,
        spatial_radius_km=spatial_radius_km,
        minutes_for_a_stop=minutes_for_a_stop,
        leaving_time=True
    )

    # 3) Mise en forme du résultat
    stops_df = stays.rename(columns={
        'datetime'         : 'start_time',
        'leaving_datetime' : 'end_time',
        'lng'              : 'lon'
    })
    # Calcul de la durée en secondes
    stops_df['duration_s'] = (
        stops_df['end_time'] - stops_df['start_time']
    ).dt.total_seconds()

    # Ne garder que les colonnes utiles
    return stops_df[['start_time', 'end_time', 'duration_s', 'lat', 'lon']]
