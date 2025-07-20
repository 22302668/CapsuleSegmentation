#!/usr/bin/env python3
import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from load_and_preprocess         import load_data_and_prepare
from movingpandas_stop_detection import detect_stops_and_moves
from dbscan_clustering           import cluster_stops_dbscan
from group_stops                 import group_stops_by_time_and_space
from classify_home_work          import classify_home_work
from evaluate_home_work          import evaluate_home_work_classification
from verify_stop_activities      import verify_stop_activities
from split_moves_stops           import tag_moves_with_stop_types
from generate_report             import generate_full_report

def generate_report_for_participant(df: pd.DataFrame, pid: str, engine) -> None:
    os.makedirs("data", exist_ok=True)
    path_html = f"data/{pid}_rapport.html"

    # 1) Détection des stops et moves
    raw_stops, raw_moves = detect_stops_and_moves(
        df,
        min_duration_minutes=5,
        max_diameter_meters=100,
        min_move_duration_s=30,
        min_time_gap_s=900
    )
    if raw_stops.empty:
        print(f"❌ Aucun stop détecté pour {pid}")
        return

    # 2) Regroupement spatial/temporal (DBSCAN)
    _, clustered_stops = cluster_stops_dbscan(
        df,
        raw_stops,
        eps_m=150,
        min_samples=1
    )
    grouped_stops = group_stops_by_time_and_space(
        clustered_stops,
        max_time_gap_s=600,
        max_distance_m=200
    )
    if grouped_stops.empty:
        print(f"❌ Aucun stop agrégé pour {pid}")
        return

    # 3) Classification Home/Work/Autre
    final_stops = classify_home_work(grouped_stops)
    evaluation  = evaluate_home_work_classification(final_stops)

    # 4) Vérification (activités externes)
    matched, _ = verify_stop_activities(final_stops, engine, pid)
    unknowns = matched[matched['place_type']=='autre'] if not matched.empty else pd.DataFrame()

    # 5) Étiquetage des moves
    moves_tagged = tag_moves_with_stop_types(raw_moves, final_stops, max_dist_m=100)

    # 6) Sauvegardes CSV
    raw_stops.to_csv(f"data/{pid}_all_stops.csv", index=False)
    moves_tagged.to_csv(f"data/{pid}_all_moves.csv", index=False)

    # 7) Génération du rapport HTML
    #    Assurez‑vous que la signature de generate_full_report est bien :
    #    (df_all, stops_summary_all, stops_summary_grouped, final_merged_stops,
    #     evaluation, unknowns, moves)
    section = generate_full_report(
        df,            # toutes les observations GPS
        raw_stops,     # stops bruts
        grouped_stops, # stops agrégés
        final_stops,   # stops classifiés Home/Work/Autre
        evaluation,    # évaluation Home/Work
        unknowns,      # stops « autre » vérifiés
        moves_tagged   # moves étiquetés
    )
    with open(path_html, 'w', encoding='utf-8') as f:
        f.write(
            '<!DOCTYPE html><html><head><meta charset="UTF-8">'
            '<title>Rapport GPS</title>'
            '<style>body{font-family:Arial; margin:20px;}'
            'h1,h2,h3{color:#2c3e50;}hr{margin:40px 0;}</style>'
            '</head><body>'
        )
        f.write(f'<h1>Rapport GPS – Participant {pid}</h1>')
        f.write(f'<p><em>Date : {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}</em></p>')
        f.write(section)
        f.write('</body></html>')

    print(f"✅ Rapport généré → {path_html}")

def main() -> None:
    load_dotenv()
    url = (
        f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}"
        f"@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DB')}"
    )
    engine = create_engine(url)

    with engine.connect() as conn:
        pids = pd.read_sql_query(
            text("SELECT DISTINCT participant_id FROM gps_all_participants"),
            conn
        )['participant_id'].tolist()

    for pid in pids:
        print(f"\n=== Participant {pid} ===")
        df = load_data_and_prepare(engine, pid)
        if df.empty:
            print("Aucun point GPS.")
            continue
        generate_report_for_participant(df, pid, engine)

if __name__ == '__main__':
    main()
