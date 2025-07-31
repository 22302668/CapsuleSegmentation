import pandas as pd
from geopy.distance import geodesic

def classify_home_work(
    stops_df: pd.DataFrame,
    home_window: tuple[int, int] = (20, 8),
    work_window: tuple[int, int] = (7, 19),
    match_radius_m: float = 100,
    round_precision: int = 3
) -> pd.DataFrame:
    
    df = stops_df.copy()
    df['start_time'] = pd.to_datetime(df['start_time'], utc=True).dt.tz_convert('Europe/Paris')
    df['end_time']   = pd.to_datetime(df['end_time'], utc=True).dt.tz_convert('Europe/Paris')
    df['lat_round'] = df['lat'].round(round_precision)
    df['lon_round'] = df['lon'].round(round_precision)
    df['place_type'] = 'autre'

    def covers_window(row, h0, h1):
        s, e = row['start_time'], row['end_time']
        hours = list(range(h0, 24)) + list(range(0, h1)) if h0 > h1 else list(range(h0, h1))
        for h in hours:
            t = s.replace(hour=h, minute=0, second=0, microsecond=0)
            if s <= t <= e:
                return True
        return False

    home_coords, work_coords = None, None

    # HOME : le + fréquent entre 20h–8h
    df['covers_home'] = df.apply(lambda r: covers_window(r, *home_window), axis=1)
    homes = df[df['covers_home']]
    if not homes.empty:
        home_zone = homes.groupby(['lat_round', 'lon_round'])['duration_s'].sum().idxmax()
        home_coords = (home_zone[0], home_zone[1])
        df['dist_to_home'] = df.apply(lambda r: geodesic((r.lat, r.lon), home_coords).meters, axis=1)
        df.loc[df['dist_to_home'] <= match_radius_m, 'place_type'] = 'Home'
        print(f"[INFO] Home détecté à {home_coords}")
    else:
        print("[INFO] Aucun stop la nuit trouvé pour détecter Home.")

    # WORK : le + fréquent entre 7h–19h, hors Home
    df['covers_work'] = df.apply(lambda r: covers_window(r, *work_window), axis=1)
    work_cand = df[(df['covers_work']) & (df['place_type'] == 'autre')]
    if not work_cand.empty:
        work_zone = work_cand.groupby(['lat_round', 'lon_round'])['duration_s'].sum().idxmax()
        work_coords = (work_zone[0], work_zone[1])
        df['dist_to_work'] = df.apply(lambda r: geodesic((r.lat, r.lon), work_coords).meters, axis=1)
        df.loc[df['dist_to_work'] <= match_radius_m, 'place_type'] = 'Work'
        print(f"[INFO] Work détecté à {work_coords}")
    else:
        print("[INFO] Aucun stop en journée trouvé pour détecter Work.")

    # Format de sortie
    df['merged_starts'] = df['start_time'].dt.strftime('%Y-%m-%d %H:%M:%S').apply(lambda s: [s])
    df['merged_ends']   = df['end_time'].dt.strftime('%Y-%m-%d %H:%M:%S').apply(lambda s: [s])

    return df[['start_time','end_time','duration_s','lat','lon','place_type','merged_starts','merged_ends']]
