import pandas as pd


def merge_stops(stops_df, epsilon_lat=0.00035, epsilon_lon=0.00045, max_time_gap_s=300):
    """
    Regroupe les arrêts spatialement proches en une bounding box.
    Si tous les points sont proches entre eux, on les fusionne
    en un seul stop centré. Sinon, chaque point reste un stop indépendant.

    Args:
        stops_df (pd.DataFrame): colonnes requises ['lat', 'lon', 'start_time', 'end_time', 'duration_s']
        epsilon_lon (float): tolérance spatiale longitudinale
        epsilon_lat (float): tolérance spatiale latitudinale

    Returns:
        pd.DataFrame: stops fusionnés ou non, avec une colonne 'group_size'
    """
    grouped_stops = []

    # On trie les données
    stops_df = stops_df.sort_values("start_time").reset_index(drop=True)

    group = []
    for _, row in stops_df.iterrows():
        # Vérification du time gap si groupe existant
        if group:
            last_end = group[-1]['end_time']
            time_gap = (row['start_time'] - last_end).total_seconds()
            if time_gap > max_time_gap_s:
                # Finaliser le groupe actuel et recommencer
                temp_df = pd.DataFrame(group)
                grouped_stops.append({
                    "start_time": temp_df['start_time'].min(),
                    "end_time": temp_df['end_time'].max(),
                    "duration_s": temp_df['duration_s'].sum(),
                    "lat": (temp_df['lat'].max() + temp_df['lat'].min()) / 2,
                    "lon": (temp_df['lon'].max() + temp_df['lon'].min()) / 2,
                    "group_size": len(temp_df)
                })
                group = []

        group.append(row)
        group_df = pd.DataFrame(group)

        # Bounding box actuelle
        lat_span = group_df['lat'].max() - group_df['lat'].min()
        lon_span = group_df['lon'].max() - group_df['lon'].min()

        if lat_span <= epsilon_lat and lon_span <= epsilon_lon:
            continue  # On continue à ajouter des points dans le groupe
        else:
            # On finalise le groupe précédent 
            last_valid_group = group[:-1]
            if last_valid_group:
                temp_df = pd.DataFrame(last_valid_group)
                grouped_stops.append({
                    "start_time": temp_df['start_time'].min(),
                    "end_time": temp_df['end_time'].max(),
                    "duration_s": temp_df['duration_s'].sum(),
                    "lat": (temp_df['lat'].max() + temp_df['lat'].min()) / 2,
                    "lon": (temp_df['lon'].max() + temp_df['lon'].min()) / 2,
                    "group_size": len(temp_df)
                })
            # Nouveau groupe avec le point en trop
            group = [row]

    # Dernier groupe
    if group:
        group_df = pd.DataFrame(group)
        lat_span = group_df['lat'].max() - group_df['lat'].min()
        lon_span = group_df['lon'].max() - group_df['lon'].min()
        if lat_span <= epsilon_lat and lon_span <= epsilon_lon:
            grouped_stops.append({
                "start_time": group_df['start_time'].min(),
                "end_time": group_df['end_time'].max(),
                "duration_s": group_df['duration_s'].sum(),
                "lat": (group_df['lat'].max() + group_df['lat'].min()) / 2,
                "lon": (group_df['lon'].max() + group_df['lon'].min()) / 2,
                "group_size": len(group_df)
            })
        else:
            # Chaque point séparément
            for _, r in group_df.iterrows():
                grouped_stops.append({
                    "start_time": r['start_time'],
                    "end_time": r['end_time'],
                    "duration_s": r['duration_s'],
                    "lat": r['lat'],
                    "lon": r['lon'],
                    "group_size": 1
                })

    return pd.DataFrame(grouped_stops)
