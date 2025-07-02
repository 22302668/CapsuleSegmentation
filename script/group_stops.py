import pandas as pd

def merge_stops(stops_df, epsilon_lat=0.00035, epsilon_lon=0.00045, max_time_gap_s=300):
    """
    Regroupe les arrêts spatialement proches en une bounding box,
    sans jamais fusionner d’arrêts de dates différentes.

    Args:
        stops_df (pd.DataFrame): colonnes requises ['lat', 'lon', 'start_time', 'end_time', 'duration_s']
        epsilon_lon (float): tolérance spatiale longitudinale
        epsilon_lat (float): tolérance spatiale latitudinale
        max_time_gap_s (int): écart temporel maximal (en s) pour rester dans le même groupe

    Returns:
        pd.DataFrame: stops fusionnés ou non, avec une colonne 'group_size'
    """
    grouped_stops = []
    # Tri chronologique
    stops_df = stops_df.sort_values("start_time").reset_index(drop=True)
    group = []

    def flush_group(g):
        """Ajoute le groupe g à grouped_stops puis vide g."""
        if not g:
            return
        dfg = pd.DataFrame(g)
        grouped_stops.append({
            "start_time": dfg['start_time'].min(),
            "end_time":   dfg['end_time'].max(),
            "duration_s": dfg['duration_s'].sum(),
            "lat":        (dfg['lat'].max()  + dfg['lat'].min())  / 2,
            "lon":        (dfg['lon'].max()  + dfg['lon'].min())  / 2,
            "group_size": len(dfg)
        })
        g.clear()

    for _, row in stops_df.iterrows():
        # 1) Si le groupe courant porte sur un autre jour -> on l'expédie
        if group and row['start_time'].date() != group[0]['start_time'].date():
            flush_group(group)

        # 2) Si écart temporel > max_time_gap_s -> on expédie aussi
        if group:
            last_end = group[-1]['end_time']
            if (row['start_time'] - last_end).total_seconds() > max_time_gap_s:
                flush_group(group)

        # 3) On ajoute toujours le nouveau point, puis on contrôle la bbox
        group.append(row)
        dfg = pd.DataFrame(group)
        lat_span = dfg['lat'].max() - dfg['lat'].min()
        lon_span = dfg['lon'].max() - dfg['lon'].min()

        # 4) Si on dépasse la tolérance spatiale, on expédie tout sauf ce dernier
        if lat_span > epsilon_lat or lon_span > epsilon_lon:
            # on expédie tout sauf row
            previous = group[:-1]
            flush_group(previous)
            # et on repart un nouveau groupe avec row seul
            group = [row]

    # À la fin, expédier ce qui reste
    flush_group(group)

    return pd.DataFrame(grouped_stops)
