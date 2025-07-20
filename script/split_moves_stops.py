import pandas as pd
from geopy.distance import geodesic
import geopandas as gpd
from shapely.geometry import Point
import logging

def split_stops_moves(
    gps_df: pd.DataFrame,
    stops_df: pd.DataFrame,
    min_move_duration_s: int = 30,
    min_time_gap_s: int = 900
):
    gps = gps_df.copy()
    gps['timestamp'] = (
        pd.to_datetime(gps['timestamp'], utc=True)
          .dt.tz_convert('Europe/Paris')
          .dt.tz_localize(None)
    )

    stops = stops_df.copy()
    for col in ['start_time', 'end_time']:
        stops[col] = (
            pd.to_datetime(stops[col], utc=True)
              .dt.tz_convert('Europe/Paris')
              .dt.tz_localize(None)
        )

    # masque des points GPS contenus dans un stop
    mask_stop = pd.Series(False, index=gps.index)
    for _, stop in stops.iterrows():
        mask_stop |= (
            (gps['timestamp'] >= stop['start_time']) &
            (gps['timestamp'] <= stop['end_time'])
        )

    ds1 = gps[mask_stop].reset_index(drop=True)
    ds2 = gps[~mask_stop].reset_index(drop=True)

    moves_summary = build_moves_summary(
        ds2,
        min_move_duration_s=min_move_duration_s,
        min_time_gap_s=min_time_gap_s
    )

    return ds1, ds2, moves_summary


def build_moves_summary(
    ds2: pd.DataFrame,
    min_move_duration_s: int = 60,
    min_time_gap_s: int = 1800
) -> pd.DataFrame:
    if ds2.empty:
        return pd.DataFrame()

    ds2 = ds2.sort_values('timestamp').reset_index(drop=True)
    ds2['move_id'] = (ds2['timestamp'].diff().dt.total_seconds() > min_time_gap_s).cumsum()

    moves = []
    for _, group in ds2.groupby('move_id'):
        if len(group) < 2:
            continue

        start_time = group['timestamp'].iloc[0]
        end_time   = group['timestamp'].iloc[-1]
        duration_s = (end_time - start_time).total_seconds()
        if duration_s < min_move_duration_s:
            continue

        lat_start, lon_start = group.iloc[0][['lat','lon']]
        lat_end,   lon_end   = group.iloc[-1][['lat','lon']]
        dist_m = geodesic((lat_start, lon_start), (lat_end, lon_end)).meters

        moves.append({
            'start_time': start_time,
            'end_time':   end_time,
            'duration_s': duration_s,
            'lat_start':  lat_start,
            'lon_start':  lon_start,
            'lat_end':    lat_end,
            'lon_end':    lon_end,
            'dist_m':     dist_m
        })

    return pd.DataFrame(moves)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def tag_moves_with_stop_types(
    moves: pd.DataFrame,
    stops: pd.DataFrame,
    max_dist_m: float = 100
) -> pd.DataFrame:
    """
    Pour chaque move, on rattache le type (Home/Work/Autre) du stop
    le plus proche en début et en fin de segment, en considérant
    uniquement les stops à moins de max_dist_m (distance euclidienne en mètres).
    """

    logger.info(f"▶ Entrée tag_moves_with_stop_types | "
                f"moves: {len(moves)} lignes, stops: {len(stops)} lignes")

    # Cas trivial : retourne tout en 'unknown'
    if moves.empty or stops.empty:
        logger.warning("⚠️ Pas de moves ou pas de stops → tous les types passent à 'unknown'")
        mv = moves.copy()
        mv['origin_type']      = 'unknown'
        mv['destination_type'] = 'unknown'
        mv['transition']       = mv['origin_type'] + ' → ' + mv['destination_type']
        logger.info(f"◀ Sortie tag_moves_with_stop_types | {len(mv)} moves étiquetés en 'unknown'")
        return mv

    # 1) Stops → GeoDataFrame en mètres
    stops = stops.copy()
    stops['stop_geom'] = stops.apply(
        lambda r: Point(r['lon'], r['lat']), axis=1
    )
    gdf_stops = gpd.GeoDataFrame(
        stops, geometry='stop_geom', crs='EPSG:4326'
    ).to_crs(epsg=3857)
    logger.info(f"• gdf_stops projeté en EPSG:3857 ({len(gdf_stops)} arrêts)")

    # 2) Moves → GeoDataFrame avec origines/destinations en mètres
    mv = moves.copy()
    mv['origin_geom'] = mv.apply(
        lambda r: Point(r['lon_origin'], r['lat_origin']), axis=1
    )
    mv['dest_geom'] = mv.apply(
        lambda r: Point(r['lon_dest'],   r['lat_dest']),   axis=1
    )
    gdf_moves = gpd.GeoDataFrame(
        mv, geometry='origin_geom', crs='EPSG:4326'
    ).to_crs(epsg=3857)
    logger.info(f"• gdf_moves projeté en EPSG:3857 ({len(gdf_moves)} déplacements)")

    # 3) utilitaire de recherche du type le plus proche
    def find_nearest_type(pt: Point) -> str:
        # calcule toutes les distances (en m)
        dists = gdf_stops.geometry.distance(pt)
        within = dists <= max_dist_m
        count = within.sum()
        logger.info(f"  - find_nearest_type: {count} candidats ≤ {max_dist_m} m")
        if count == 0:
            return 'unknown'
        idx = dists[within].idxmin()
        tp = gdf_stops.at[idx, 'place_type']
        logger.info(f"    → nearest idx={idx}, type={tp}, dist={dists[idx]:.1f} m")
        return tp

    # 4) Étiqueter origin_type
    gdf_moves['origin_type'] = gdf_moves.geometry.apply(find_nearest_type)
    logger.info("• origin_type étiquetés")

    # 5) Passer à dest_geom + étiqueter destination_type
    gdf_moves = gdf_moves.set_geometry('dest_geom')
    gdf_moves['destination_type'] = gdf_moves.geometry.apply(find_nearest_type)
    logger.info("• destination_type étiquetés")

    # 6) Nettoyage et sortie
    res = gdf_moves.to_crs(epsg=4326).copy()
    res = res.drop(columns=['origin_geom', 'dest_geom'])
    res['transition'] = res['origin_type'] + ' → ' + res['destination_type']

    logger.info(f"◀ Sortie tag_moves_with_stop_types | {len(res)} moves étiquetés")
    return res.reset_index(drop=True)
