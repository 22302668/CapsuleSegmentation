# movingpandas_stop_detection.py

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import movingpandas as mpd
from datetime import timedelta

def detect_stops_and_moves(
    df: pd.DataFrame,
    min_duration_minutes: int = 5,
    max_diameter_meters: float = 100,
    min_move_duration_s: float = 30,
    min_time_gap_s: float = 900
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Détecte les stops et les moves avec MovingPandas.

    Returns:
        raw_stops: DataFrame des stops bruts [start_time,end_time,duration_s,lat,lon]
        moves:     DataFrame des moves [start_time,end_time,duration_s,
                                       lat_origin,lon_origin,lat_dest,lon_dest]
    """

    # --- 0) drop tz pour MovingPandas ---
    df = df.copy()
    if df['timestamp'].dt.tz is not None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)

    # --- 1) GeoDataFrame ---
    df['geometry'] = df.apply(lambda r: Point(r['lon'], r['lat']), axis=1)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')

    # --- 2) Trajectoire & détecteur ---
    traj = mpd.Trajectory(gdf, traj_id=1, t='timestamp')
    detector = mpd.TrajectoryStopDetector(traj)

    # --- 3) Stops bruts ---
    stops = detector.get_stop_points(
        min_duration=timedelta(minutes=min_duration_minutes),
        max_diameter=max_diameter_meters
    )
    if stops.empty:
        return pd.DataFrame(), pd.DataFrame()

    # renommer dynamiquement les colonnes
    start_col = 't0' if 't0' in stops.columns else 'start_time'
    end_col   = 't1' if 't1' in stops.columns else 'end_time'
    stops = stops.rename(columns={
        start_col:  'start_time',
        end_col:    'end_time',
        'geometry': 'stop_geom'
    })
    stops['duration_s'] = (stops['end_time'] - stops['start_time']).dt.total_seconds()
    stops['lat'] = stops['stop_geom'].y
    stops['lon'] = stops['stop_geom'].x

    raw_stops = stops[[
        'start_time','end_time','duration_s','lat','lon'
    ]].sort_values('start_time').reset_index(drop=True)

    # --- 4) Moves entre stops ---
    moves = []
    prev_end = df['timestamp'].min()
    for _, stop in raw_stops.iterrows():
        window = df[
            (df['timestamp'] >= prev_end) &
            (df['timestamp'] <= stop['start_time'])
        ]
        if len(window) >= 2:
            dt  = (window['timestamp'].iloc[-1] - window['timestamp'].iloc[0]).total_seconds()
            gap = (stop['start_time'] - prev_end).total_seconds()
            if dt >= min_move_duration_s and gap >= min_time_gap_s:
                moves.append({
                    'start_time':  window['timestamp'].iloc[0],
                    'end_time':    window['timestamp'].iloc[-1],
                    'duration_s':  dt,
                    'lat_origin':  window['lat'].iloc[0],
                    'lon_origin':  window['lon'].iloc[0],
                    'lat_dest':    window['lat'].iloc[-1],
                    'lon_dest':    window['lon'].iloc[-1],
                })
        prev_end = stop['end_time']

    # move après le dernier stop
    window = df[df['timestamp'] >= prev_end]
    if len(window) >= 2:
        dt = (window['timestamp'].iloc[-1] - window['timestamp'].iloc[0]).total_seconds()
        if dt >= min_move_duration_s:
            moves.append({
                'start_time': window['timestamp'].iloc[0],
                'end_time':   window['timestamp'].iloc[-1],
                'duration_s': dt,
                'lat_origin': window['lat'].iloc[0],
                'lon_origin': window['lon'].iloc[0],
                'lat_dest':   window['lat'].iloc[-1],
                'lon_dest':   window['lon'].iloc[-1],
            })

    moves_df = pd.DataFrame(moves)
    return raw_stops, moves_df
