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

logger = logging.getLogger(__name__)

def tag_moves_with_stop_types(
    moves: pd.DataFrame,
    stops: pd.DataFrame,
    max_dist_m: float = 200
) -> pd.DataFrame:
    """
    Pour chaque déplacement (move), assigne un type d'origine et de destination
    en fonction du stop le plus proche.

    Args:
        moves (pd.DataFrame): DataFrame avec colonnes ['lat_origin','lon_origin','lat_dest','lon_dest']
        stops (pd.DataFrame): DataFrame des arrêts avec colonnes ['lat','lon','place_type']
        max_dist_m (float): distance max pour rattacher un move à un stop (mètres)

    Returns:
        pd.DataFrame: moves enrichi de colonnes ['origin_type','destination_type']
    """
    logger.info(f"tag_moves_with_stop_types: {len(moves)} moves, {len(stops)} stops, max_dist_m={max_dist_m}")

    # 1) Construire GeoDataFrame des stops
    gdf_stops = gpd.GeoDataFrame(
        stops.copy(),
        geometry=gpd.points_from_xy(stops['lon'], stops['lat']),
        crs='EPSG:4326'
    )
    logger.debug("GeoDataFrame stops créé avec CRS EPSG:4326")

    # 2) Fonction d'assignation de type pour un point
    def nearest_place(lat, lon):
        point = Point(lon, lat)
        # calculer distance à chaque stop
        gdf_stops['dist'] = gdf_stops.geometry.apply(
            lambda s: geodesic((lat, lon), (s.y, s.x)).meters
        )
        # logging des 5 plus proches pour debug si besoin
        nearest_five = gdf_stops.nsmallest(5, 'dist')[['place_type', 'dist']]
        logger.debug(f"nearest_place({lat:.6f},{lon:.6f}) → 5 closest:\n{nearest_five}")

        nearest = gdf_stops.loc[gdf_stops['dist'].idxmin()]
        if nearest['dist'] <= max_dist_m:
            logger.info(f"Point ({lat:.6f},{lon:.6f}) rattaché à '{nearest['place_type']}' (dist={nearest['dist']:.1f} m)")
            return nearest['place_type']
        else:
            logger.info(f"Point ({lat:.6f},{lon:.6f}) pas de stop <{max_dist_m} m (min_dist={nearest['dist']:.1f} m) → unknown")
            return 'unknown'

    # 3) Appliquer pour origine et destination
    moves = moves.copy()
    if moves.empty:
        logger.warning("moves vide, je rajoute directly 'unknown' pour origin_type et destination_type")
        moves['origin_type'] = 'unknown'
        moves['destination_type'] = 'unknown'
    else:
        logger.info("Étiquetage origin_type")
        moves['origin_type'] = moves.apply(
            lambda r: nearest_place(r['lat_origin'], r['lon_origin']),
            axis=1
        )
        logger.info("Étiquetage destination_type")
        moves['destination_type'] = moves.apply(
            lambda r: nearest_place(r['lat_dest'], r['lon_dest']),
            axis=1
        )

    # 4) Transition
    moves['transition'] = moves['origin_type'] + ' → ' + moves['destination_type']
    logger.info("Transitions construites")

    # # 5) Calculer des distances selon les types
    # def calc_distance_with_type(row, type1):
    #     """Calcule la distance si au moins un des deux lieux est type1 (Home ou Work)."""
    #     if type1 in {row.origin_type, row.destination_type}:
    #         return geodesic(
    #             (row.lat_origin, row.lon_origin),
    #             (row.lat_dest,   row.lon_dest)
    #         ).meters
    #     return None

    # moves['dist_from_home'] = moves.apply(lambda r: calc_distance_with_type(r, 'Home'), axis=1)
    # moves['dist_from_work'] = moves.apply(lambda r: calc_distance_with_type(r, 'Work'), axis=1)

    return moves.reset_index(drop=True)

def snap_moves_to_home_work(moves, final_stops, max_dist_m=150):
    """
    Crée une copie des moves et remplace les coordonnées des points proches de Home/Work
    par celles de Home/Work, sans modifier le DataFrame original.
    Ajoute :
        - snapped (bool) : True si recalé
        - snapped_origin_type / snapped_destination_type : Home/Work si recalé
    """
    moves_copy = moves.copy()
    moves_copy['snapped'] = False
    moves_copy['snapped_origin_type'] = None
    moves_copy['snapped_destination_type'] = None

    # Récupération des coordonnées
    if 'Home' not in final_stops['place_type'].values or 'Work' not in final_stops['place_type'].values:
        return moves_copy

    home = final_stops.loc[final_stops['place_type'] == 'Home', ['lat', 'lon']].iloc[0].to_dict()
    work = final_stops.loc[final_stops['place_type'] == 'Work', ['lat', 'lon']].iloc[0].to_dict()

    for idx, row in moves_copy.iterrows():
        # Origine
        if geodesic((row.lat_origin, row.lon_origin), (home['lat'], home['lon'])).meters <= max_dist_m:
            moves_copy.at[idx, 'lat_origin'] = home['lat']
            moves_copy.at[idx, 'lon_origin'] = home['lon']
            moves_copy.at[idx, 'snapped'] = True
            moves_copy.at[idx, 'snapped_origin_type'] = 'Home'
        elif geodesic((row.lat_origin, row.lon_origin), (work['lat'], work['lon'])).meters <= max_dist_m:
            moves_copy.at[idx, 'lat_origin'] = work['lat']
            moves_copy.at[idx, 'lon_origin'] = work['lon']
            moves_copy.at[idx, 'snapped'] = True
            moves_copy.at[idx, 'snapped_origin_type'] = 'Work'

        # Destination
        if geodesic((row.lat_dest, row.lon_dest), (home['lat'], home['lon'])).meters <= max_dist_m:
            moves_copy.at[idx, 'lat_dest'] = home['lat']
            moves_copy.at[idx, 'lon_dest'] = home['lon']
            moves_copy.at[idx, 'snapped'] = True
            moves_copy.at[idx, 'snapped_destination_type'] = 'Home'
        elif geodesic((row.lat_dest, row.lon_dest), (work['lat'], work['lon'])).meters <= max_dist_m:
            moves_copy.at[idx, 'lat_dest'] = work['lat']
            moves_copy.at[idx, 'lon_dest'] = work['lon']
            moves_copy.at[idx, 'snapped'] = True
            moves_copy.at[idx, 'snapped_destination_type'] = 'Work'

    return moves_copy