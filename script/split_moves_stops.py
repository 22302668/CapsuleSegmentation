# fichier : split_moves_stops.py

import pandas as pd

def split_stops_moves(gps_df: pd.DataFrame, stops_df: pd.DataFrame):
    """
    Sépare gps_df en deux DataFrames :
     - ds1 : tous les points dont timestamp est dans un arrêt (stops_df)
     - ds2 : le reste (les moves)
    """
    # Copie et normalisation des timestamps
    gps = gps_df.copy()
    gps['timestamp'] = pd.to_datetime(gps['timestamp'])
    # Si tz-aware, on retire la timezone pour uniformiser
    if gps['timestamp'].dt.tz is not None:
        gps['timestamp'] = gps['timestamp'].dt.tz_localize(None)

    stops = stops_df.copy()
    stops['start_time'] = pd.to_datetime(stops['start_time'])
    stops['end_time']   = pd.to_datetime(stops['end_time'])
    # idem pour les stops
    if hasattr(stops['start_time'].dt, 'tz'):
        stops['start_time'] = stops['start_time'].dt.tz_localize(None)
        stops['end_time']   = stops['end_time'].dt.tz_localize(None)

    # on construit un masque "dans un stop"
    mask_stop = pd.Series(False, index=gps.index)
    for _, stop in stops.iterrows():
        mask_stop |= (
            (gps['timestamp'] >= stop['start_time']) &
            (gps['timestamp'] <= stop['end_time'])
        )

    ds1 = gps[mask_stop].reset_index(drop=True)
    ds2 = gps[~mask_stop].reset_index(drop=True)
    return ds1, ds2
