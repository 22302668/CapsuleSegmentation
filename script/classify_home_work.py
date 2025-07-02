import pandas as pd
from geopy.distance import geodesic

def classify_home_work(
    stops_df: pd.DataFrame,
    home_window: tuple[int, int] = (2, 5),  
    work_window: tuple[int, int] = (9, 18),
    match_radius_m: float = 150,
    round_precision: int = 3
) -> pd.DataFrame:
    """
    Classifie les lieux en 'Home', 'Work' ou 'autre'.
    - Home : zones couvrant au moins une heure entre home_window (minuit–6 h).
    - Work : zones couvrant au moins une heure entre work_window (9–18 h), hors Home.
    - Autre : le reste.
    """
    df = stops_df.copy()
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time']   = pd.to_datetime(df['end_time'])

    # 1) Zones arrondies
    df['lat_round'] = df['lat'].round(round_precision)
    df['lon_round'] = df['lon'].round(round_precision)
    df['place_type'] = 'autre'

    # Helper: test si le stop chevauche au moins une heure de [h0,h1]
    def covers_window(row, h0, h1):
        s, e = row['start_time'], row['end_time']
        # pour chaque heure entière de h0 à h1 incluse
        for h in range(h0, h1+1):
            cand = s.replace(hour=h, minute=0, second=0, microsecond=0)
            # si le stop traverse minuit on teste aussi sur end_time
            cand2 = e.replace(hour=h, minute=0, second=0, microsecond=0)
            if (s <= cand <= e) or (s <= cand2 <= e):
                return True
        return False

    # 2) Détection du Home entre 0 h et 6 h
    df['covers_home'] = df.apply(lambda r: covers_window(r, *home_window), axis=1)
    homes = df[df['covers_home']]
    if not homes.empty:
        # zone arrondie de durée max
        home_zone = homes.groupby(['lat_round','lon_round'])['duration_s'].sum().idxmax()
        home_coords = (home_zone[0], home_zone[1])
        # tous les stops à <= match_radius_m deviennent Home
        mask_home = df['covers_home'] & df.apply(
            lambda r: geodesic((r.lat, r.lon), home_coords).meters <= match_radius_m,
            axis=1
        )
        df.loc[mask_home, 'place_type'] = 'Home'

    # 3) Détection du Work entre work_window, parmi ceux encore 'autre'
    df['covers_work'] = df.apply(lambda r: covers_window(r, *work_window), axis=1)
    work_cand = df[(df['covers_work']) & (df['place_type'] == 'autre')]
    if not work_cand.empty:
        work_zone = work_cand.groupby(['lat_round','lon_round'])['duration_s'].sum().idxmax()
        work_coords = (work_zone[0], work_zone[1])
        mask_work = (df['place_type'] == 'autre') & df.apply(
            lambda r: geodesic((r.lat, r.lon), work_coords).meters <= match_radius_m,
            axis=1
        )
        df.loc[mask_work, 'place_type'] = 'Work'

    # Pour que evaluate_home_work & plot_combined_confidence_score fonctionnent
    df['merged_starts'] = df['start_time'] \
        .dt.strftime('%Y-%m-%d %H:%M:%S') \
        .apply(lambda s: [s])
    df['merged_ends']   = df['end_time'] \
        .dt.strftime('%Y-%m-%d %H:%M:%S') \
        .apply(lambda s: [s])

    return df[['start_time','end_time','duration_s','lat','lon','place_type',
          'merged_starts','merged_ends']]
