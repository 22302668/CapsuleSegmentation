# benchmark_stops.py (corrigé avec paramètre engine)
import time
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

from stop_detection.staypoint_distance_duration import detect_stops as method_staypoint
from stop_detection.speed_duration_threshold import detect_stops as method_speed
from stop_detection.hdbscan_stop_filtering import detect_stops as method_hdbscan
from stop_detection.scikitmobility_staypoints import detect_stops as method_skmob
from load_and_preprocess import load_data_and_prepare
from clustering_hdbscan import cluster_and_visualize

def benchmark_methods(df):
    methods = {
        'staypoint_distance': method_staypoint,
        'speed_duration': method_speed,
        'hdbscan_noise': method_hdbscan,
        'scikitmobility': method_skmob
    }

    results = []

    for name, func in methods.items():
        try:
            start = time.time()
            stops = func(df.copy(deep=True)) if isinstance(df, pd.DataFrame) else func(df[0].copy(deep=True))
            end = time.time()

            results.append({
                'method': name,
                'n_stops': len(stops),
                'mean_duration_s': stops['duration_s'].mean() if 'duration_s' in stops.columns else None,
                'total_duration_s': stops['duration_s'].sum() if 'duration_s' in stops.columns else None,
                'execution_time_s': round(end - start, 3)
            })
        except Exception as e:
            results.append({
                'method': name,
                'n_stops': None,
                'mean_duration_s': None,
                'total_duration_s': None,
                'execution_time_s': None,
                'error': str(e)
            })

    return pd.DataFrame(results)

if __name__ == "__main__":
    load_dotenv()
    url = f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DB')}"
    engine = create_engine(url)

    print("\n🚀 Chargement des données...")
    data = load_data_and_prepare(engine)
    df = data if isinstance(data, pd.DataFrame) else data[0]
    df = cluster_and_visualize(df)  # ajoute la colonne 'cluster_behavior'
    print("📊 Stats des données :")
    print("Points totaux :", len(df))
    print("Min speed:", df['speed_kmh'].min())
    print("Max speed:", df['speed_kmh'].max())
    print("Min durée entre points:", df['time_diff_s'].min())
    print("Max durée entre points:", df['time_diff_s'].max())
    print("Clusters présents:", df['cluster_behavior'].unique() if 'cluster_behavior' in df.columns else "Aucun")

    df_resultats = benchmark_methods(df)
    print("\n📊 Résultats du benchmark :")
    print(df_resultats)
    df_resultats.to_csv("data/benchmark_stop_detection.csv", index=False)