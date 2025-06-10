import pandas as pd
from geopy.distance import geodesic
from datetime import time, datetime

def classify_home_work(
    stops_df,
    home_hour_target=3,
    work_hour_target=11,
    match_radius_m=150,
    round_precision=3
):
    """
    Classifie les lieux en 'Home', 'Work' ou 'autre' selon l’heure exacte à laquelle un stop est actif.
    Le lieu couvrant le plus souvent 3 h du matin est considéré comme 'Home',
    celui couvrant le plus souvent 11 h est considéré comme 'Work'.
    
    Args:
        stops_df (pd.DataFrame): Doit contenir au moins les colonnes
            ['start_time', 'end_time', 'duration_s', 'lat', 'lon'].
        home_hour_target (int): Heure cible pour détecter le lieu 'Home' (ex: 3).
        work_hour_target (int): Heure cible pour détecter le lieu 'Work' (ex: 11).
        match_radius_m (float): Rayon (en mètres) pour associer d’autres arrêts au même lieu.
        round_precision (int): Précision pour regrouper les lieux GPS (lat/lon roundés).
    
    Returns:
        pd.DataFrame: Lieux fusionnés avec étiquettes 'Home', 'Work' ou 'autre', regroupés par zone arrondie.
    """
    df = stops_df.copy()
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time']   = pd.to_datetime(df['end_time'])

    # Arrondir légèrement lat/lon pour créer des “zones” (~100 m si round_precision=3)
    df['lat_round'] = df['lat'].round(round_precision)
    df['lon_round'] = df['lon'].round(round_precision)
    df['place_type'] = 'autre'

    def covers_exact_hour(row, target_hour):
        """
        Retourne True si, pour la date de start_time, le timestamp “target_hour:00:00”
        se trouve dans l’intervalle [start_time, end_time], même si le stop traverse minuit.
        """
        start_dt = row['start_time']
        end_dt   = row['end_time']

        # Candidate sur la même date que start_dt
        cand_same_day = start_dt.replace(hour=target_hour, minute=0, second=0, microsecond=0)

        # Si le stop ne traverse pas minuit (start_date == end_date)
        if start_dt.date() == end_dt.date():
            return (start_dt <= cand_same_day) and (cand_same_day <= end_dt)
        else:
            # Si le stop traverse minuit, on teste aussi la date de end_dt
            cand_next_day = end_dt.replace(hour=target_hour, minute=0, second=0, microsecond=0)
            return ((start_dt <= cand_same_day) and (cand_same_day <= end_dt)) \
                or ((start_dt <= cand_next_day) and (cand_next_day <= end_dt))

    # Détection du lieu Home (3 h)
    df['covers_home_hour'] = df.apply(lambda r: covers_exact_hour(r, home_hour_target), axis=1)
    df_home = df[df['covers_home_hour']]
    if not df_home.empty:
        # On sélectionne la “zone arrondie” (lat_round, lon_round) qui cumule la durée la plus longue
        home_duration = df_home.groupby(['lat_round', 'lon_round'])['duration_s'].sum()
        home_location = home_duration.idxmax()            # e.g. (48.849, 2.383)
        home_coords   = (home_location[0], home_location[1])
        print(f"➔ Home détecté vers {home_location}")
        # On marque “Home” pour tous les arrêts (qui couvrent 3 h) à moins de match_radius_m du centre
        for idx, row in df[df['covers_home_hour']].iterrows():
            if geodesic((row['lat'], row['lon']), home_coords).meters <= match_radius_m:
                df.at[idx, 'place_type'] = 'Home'

    # Détection du lieu Work (11 h), on exclut ceux déjà marqués “Home”
    df['covers_work_hour'] = df.apply(lambda r: covers_exact_hour(r, work_hour_target), axis=1)
    df_work_cand = df[(df['covers_work_hour']) & (df['place_type'] == 'autre')]
    if not df_work_cand.empty:
        work_duration = df_work_cand.groupby(['lat_round', 'lon_round'])['duration_s'].sum()
        work_location = work_duration.idxmax()
        work_coords   = (work_location[0], work_location[1])
        print(f"➔ Work détecté vers {work_location}")
        # On marque “Work” pour tous les arrêts “autre” proches de work_coords
        for idx, row in df[df['place_type'] == 'autre'].iterrows():
            if geodesic((row['lat'], row['lon']), work_coords).meters <= match_radius_m:
                df.at[idx, 'place_type'] = 'Work'

    # Regroupement final par zone arrondie
    grouped = (
        df.groupby(['lat_round', 'lon_round', 'place_type'])
          .agg(
              start_time   = ('start_time', 'min'),
              end_time     = ('end_time', 'max'),
              duration_s   = ('duration_s', 'sum'),
              lat          = ('lat', 'mean'),
              lon          = ('lon', 'mean'),
              group_size   = ('duration_s', 'count'),
              merged_starts = ('start_time', lambda x: list(x.dt.strftime('%Y-%m-%d %H:%M:%S'))),
              merged_ends   = ('end_time',   lambda x: list(x.dt.strftime('%Y-%m-%d %H:%M:%S')))
          )
          .reset_index()
    )

    return grouped
