from geopy.distance import geodesic
import pandas as pd

def group_stops_by_time_and_space(stops_df, max_time_gap_s=600, max_distance_m=200):
    """
    Regroupe les arrêts proches dans le temps ET dans l'espace.

    Args:
        stops_df (pd.DataFrame): Doit contenir ['start_time', 'end_time', 'duration_s', 'lat', 'lon']
        max_time_gap_s (int): Ecart temporel max (en secondes) pour fusionner
        max_distance_m (float): Distance max (en mètres) pour fusionner spatialement

    Returns:
        pd.DataFrame: Arrêts fusionnés (start_time, end_time, lat, lon, etc.)
    """
    if stops_df.empty:
        return pd.DataFrame(columns=['start_time', 'end_time', 'duration_s', 'lat', 'lon', 'group_size'])

    stops_df = stops_df.sort_values('start_time').reset_index(drop=True)
    grouped = []
    current_group = [stops_df.iloc[0]]

    for i in range(1, len(stops_df)):
        prev = current_group[-1]
        curr = stops_df.iloc[i]
        time_gap = (curr['start_time'] - prev['end_time']).total_seconds()
        distance = geodesic((prev['lat'], prev['lon']), (curr['lat'], curr['lon'])).meters

        if time_gap <= max_time_gap_s and distance <= max_distance_m:
            current_group.append(curr)
        else:
            df = pd.DataFrame(current_group)
            grouped.append({
                'start_time': df['start_time'].min(),
                'end_time': df['end_time'].max(),
                'duration_s': df['duration_s'].sum(),
                'lat': df['lat'].mean(),
                'lon': df['lon'].mean(),
                'group_size': len(df)
            })
            current_group = [curr]

    # Ajouter le dernier groupe
    df = pd.DataFrame(current_group)
    grouped.append({
        'start_time': df['start_time'].min(),
        'end_time': df['end_time'].max(),
        'duration_s': df['duration_s'].sum(),
        'lat': df['lat'].mean(),
        'lon': df['lon'].mean(),
        'group_size': len(df)
    })

    return pd.DataFrame(grouped)
