import pandas as pd

def classify_home_work(
    stops_df, home_hours=(0, 6),
    min_duration_home_s=3600,
    min_duration_work_s=900,
    round_precision=3
):
    df = stops_df.copy()
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    df['hour'] = df['start_time'].dt.hour

    # Regrouper les lieux proches
    df['lat_round'] = df['lat'].round(round_precision)
    df['lon_round'] = df['lon'].round(round_precision)

    # Marquer les arrêts de nuit pour HOME
    df['is_home_time'] = df['hour'].between(home_hours[0], home_hours[1], inclusive="left")

    # Détection du lieu Home (max durée cumulée la nuit)
    df['place_type'] = 'Unknown'
    home_durations = df[df['is_home_time']].groupby(['lat_round', 'lon_round'])['duration_s'].sum()
    if not home_durations.empty and home_durations.max() >= min_duration_home_s:
        home_location = home_durations.idxmax()
        df.loc[
            (df['lat_round'] == home_location[0]) & (df['lon_round'] == home_location[1]),
            'place_type'
        ] = 'Home'

    # Détection du lieu Work hors Home
    df_non_home = df[df['place_type'] != 'Home']
    # Choisir le lieu non-Home avec la plus grande durée totale
    durations = df_non_home.groupby(['lat_round', 'lon_round'])['duration_s'].sum()
    if not durations.empty:
        work_location = durations.idxmax()
        if durations.max() >= min_duration_work_s:
            df.loc[
                (df['lat_round'] == work_location[0]) & (df['lon_round'] == work_location[1]),
                'place_type'
            ] = 'Work'


    # 🧾 Agrégation finale (sans regroupement temporel pour éviter durée globale trop longue)

    grouped = (
        df.groupby(['lat_round', 'lon_round', 'place_type'])
            .agg(
                start_time=('start_time', 'min'),
                end_time=('end_time', 'max'),
                duration_s=('duration_s', 'sum'),
                lat=('lat', 'mean'),
                lon=('lon', 'mean'),
                group_size=('duration_s', 'count'),
                merged_intervals=('start_time', lambda x: list(x.dt.strftime('%Y-%m-%d %H:%M:%S'))),
                merged_ends=('end_time', lambda x: list(x.dt.strftime('%Y-%m-%d %H:%M:%S')))
            )
            .reset_index()
    )


    return grouped
