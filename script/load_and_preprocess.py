import pandas as pd
import numpy as np
from geopy.distance import geodesic
import os
from sqlalchemy import text
import geopandas as gpd
from skmob import TrajDataFrame
from skmob.preprocessing import filtering

def load_data_and_prepare(engine, participant_id, max_speed_kmh=150):
    """
    Charge les points GPS depuis PostgreSQL, calcule les distances, vitesses et
    lisse la vitesse, puis filtre tous les points où la vitesse instantanée
    dépasse max_speed_kmh.
    """
    with engine.connect() as conn:
        df = pd.read_sql_query(
            text("SELECT * FROM gps_all_participants WHERE participant_id = :pid"),
            con=conn,
            params={"pid": participant_id}
        )

    # 1) Horodatage et fuseau
    df['timestamp'] = (
        pd.to_datetime(df['timestamp'], utc=True)
          .dt.tz_convert('Europe/Paris')
    )
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 2) Différences temporelles
    df['time_diff_s'] = df['timestamp'].diff().dt.total_seconds()

    # 3) Calcul des distances
    distances = [np.nan]
    for i in range(1, len(df)):
        p1 = (df.loc[i-1, 'lat'], df.loc[i-1, 'lon'])
        p2 = (df.loc[i,   'lat'], df.loc[i,   'lon'])
        distances.append(geodesic(p1, p2).meters)
    df['dist_m'] = distances

    # 4) Vitesses instantanées (km/h)
    df['speed_kmh'] = df['dist_m'] / df['time_diff_s'] * 3.6
    df['speed_kmh'] = df['speed_kmh'].replace([np.inf, -np.inf], np.nan)

    # 5) Filtrage des vitesses aberrantes
    df = df[(df['speed_kmh'].isna()) | (df['speed_kmh'] <= max_speed_kmh)]

    # 6) Lissage
    df['speed_kmh_smooth'] = (
        df['speed_kmh']
          .rolling(window=5, min_periods=1, center=True)
          .mean()
    )

    return df

# def segment_by_data_weeks(df):
#     """
#     A partir d'un DataFrame contenant au moins la colonne 'timestamp' (datetime),
#     détecte automatiquement les plages de dates consécutives où il y a au moins
#     un point GPS. Renvoie une liste de tuples (date_debut, date_fin).
#     """
#     unique_dates = sorted(df['timestamp'].dt.date.unique())
#     if not unique_dates:
#         return []

#     segments = []
#     start = unique_dates[0]
#     prev = unique_dates[0]
#     for current in unique_dates[1:]:
#         if (current - prev).days > 1:
#             segments.append((start, prev))
#             start = current
#         prev = current
#     segments.append((start, prev))
#     return segments

# def clean_gps_data_full(df: pd.DataFrame, max_speed_kmh=200, min_dist_m=1, min_time_s=30, log_large_gaps=True) -> pd.DataFrame:
#     """
#     Nettoyage complet des données GPS :
#     - Supprime les doublons (lat, lon, timestamp [, participant])
#     - Supprime les points à vitesse irréaliste (> max_speed_kmh)
#     - Supprime les points très proches dans l'espace (< min_dist_m) ou le temps (< min_time_s)
#     - Lisse la vitesse calculée
#     - Alerte sur les grands trous temporels si log_large_gaps=True
#     """
#     df = df.copy()

#     # 🔧 Forcer timestamp valide
#     df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
#     df = df.dropna(subset=['timestamp', 'lat', 'lon'])

#     # ✅ Supprimer les doublons
#     subset_cols = ['timestamp', 'lat', 'lon']
#     if 'participant' in df.columns:
#         subset_cols = ['participant'] + subset_cols
#     df = df.drop_duplicates(subset=subset_cols)

#     # ✅ Renommage skmob
#     df = df.rename(columns={'timestamp': 'datetime', 'lon': 'lng'})
#     tdf = TrajDataFrame(df, timestamp=True)
#     tdf = filtering.filter(tdf, max_speed_kmh=max_speed_kmh)
#     df = pd.DataFrame(tdf).rename(columns={'datetime': 'timestamp', 'lng': 'lon'})

#     # ✅ Tri
#     df = df.sort_values('timestamp').reset_index(drop=True)

#     # ✅ Calcul du temps entre points
#     df['time_diff_s'] = df['timestamp'].diff().dt.total_seconds()
#     df.loc[df['time_diff_s'] < 0, 'time_diff_s'] = None

#     # ✅ Calcul distance géographique
#     df['dist_m'] = np.nan
#     for i in range(1, len(df)):
#         try:
#             df.at[i, 'dist_m'] = geodesic(
#                 (df.at[i-1, 'lat'], df.at[i-1, 'lon']),
#                 (df.at[i, 'lat'], df.at[i, 'lon'])
#             ).meters
#         except:
#             df.at[i, 'dist_m'] = np.nan

#     # ✅ Suppression points trop proches
#     df = df[(df['time_diff_s'].isna()) | (df['time_diff_s'] >= min_time_s)]
#     df = df[(df['dist_m'].isna()) | (df['dist_m'] >= min_dist_m)]

#     # ✅ Recalcul vitesse
#     df['speed_kmh'] = df['dist_m'] / df['time_diff_s'] * 3.6
#     df['speed_kmh'] = df['speed_kmh'].replace([np.inf, -np.inf], np.nan)
#     df['speed_kmh_smooth'] = df['speed_kmh'].rolling(window=5, min_periods=1, center=True).mean()

#     # ⚠️ Log sur grands trous temporels
#     if log_large_gaps:
#         large_gaps = df[df['time_diff_s'] > 7200]  # > 2 heures
#         if not large_gaps.empty:
#             print(f"[ALERT] {len(large_gaps)} trou(s) détecté(s) > 2h entre deux points GPS.")

#     return df.reset_index(drop=True)

# def clean_gps_data_light(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Nettoyage léger des données GPS (préserve les petits déplacements).
#     - Supprime doublons
#     - Trie les timestamps
#     - Corrige les timestamps invalides
#     """
#     df = df.copy()
#     df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
#     df = df.dropna(subset=['timestamp', 'lat', 'lon'])

#     subset_cols = ['timestamp', 'lat', 'lon']
#     if 'participant' in df.columns:
#         subset_cols = ['participant'] + subset_cols
#     df = df.drop_duplicates(subset=subset_cols)

#     df = df.sort_values('timestamp').reset_index(drop=True)
#     return df
