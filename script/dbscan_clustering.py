import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from typing import Tuple

def cluster_stops_dbscan(
    gps_df: pd.DataFrame,
    stops_df: pd.DataFrame,
    eps_m: float = 150,
    min_samples: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    gps_df   : DataFrame GPS complet (avec 'timestamp' tz-aware)
    stops_df : DataFrame des stops bruts (start_time/end_time tz-naive ou tz-aware)
    """

    # 1) Aligne les timezones des stops sur celles du gps_df
    tz = gps_df['timestamp'].dt.tz
    stops = stops_df.copy()
    stops['start_time'] = pd.to_datetime(stops['start_time'])
    stops['end_time']   = pd.to_datetime(stops['end_time'])

    # Si tz-naive, localize ; sinon convert
    if stops['start_time'].dt.tz is None:
        stops['start_time'] = stops['start_time'].dt.tz_localize(tz)
    else:
        stops['start_time'] = stops['start_time'].dt.tz_convert(tz)

    if stops['end_time'].dt.tz is None:
        stops['end_time'] = stops['end_time'].dt.tz_localize(tz)
    else:
        stops['end_time'] = stops['end_time'].dt.tz_convert(tz)

    # 2) DBSCAN spatial en haversine
    coords = np.radians(stops[['lat','lon']].to_numpy())
    kms_per_radian = 6371.0088
    epsilon = eps_m / 1000.0 / kms_per_radian

    db = DBSCAN(eps=epsilon, min_samples=min_samples, metric='haversine').fit(coords)
    stops['cluster'] = db.labels_

    # Filtre du bruit
    stops = stops[stops['cluster'] >= 0]

    # 3) Agrégation par cluster
    agg = stops.groupby('cluster').agg(
        start_time = ('start_time', 'min'),
        end_time   = ('end_time',   'max'),
        duration_s = ('duration_s','sum'),
        lat        = ('lat',       'mean'),
        lon        = ('lon',       'mean'),
        group_size = ('cluster',   'count')
    ).reset_index(drop=True)

    # 4) ds1 : points GPS correspondant à ces arrêts
    mask = pd.Series(False, index=gps_df.index)
    for _, row in agg.iterrows():
        mask |= (gps_df['timestamp'] >= row['start_time']) & (gps_df['timestamp'] <= row['end_time'])
    ds1 = gps_df[mask].copy()

    # 5) ds2 : résumé à passer au rapport
    ds2 = agg

    return ds1, ds2
