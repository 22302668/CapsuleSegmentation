import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from geopy.distance import geodesic

from load_and_preprocess           import load_data_and_prepare
from movingpandas_stop_detection  import detect_stops_and_moves
from dbscan_clustering            import cluster_stops_dbscan
from group_stops                  import group_stops_by_time_and_space
from classify_home_work           import classify_home_work
from evaluate_home_work           import evaluate_home_work_classification
from verify_stop_activities       import verify_stop_activities
from split_moves_stops            import tag_moves_with_stop_types,snap_moves_to_home_work
from generate_report              import generate_full_report

def generate_report_for_participant(df: pd.DataFrame, pid: str, engine) -> None:
    os.makedirs("data", exist_ok=True)
    path_html = f"data/{pid}_rapport.html"

    # 1+2) Détection brute des stops & moves
    raw_stops, moves = detect_stops_and_moves(
        df,
        min_duration_minutes=5,
        max_diameter_meters=100,
        min_move_duration_s=30,
        min_time_gap_s=900
    )
    if raw_stops.empty:
        print(f"Aucun stop détecté pour {pid}")
        return

    # 3) Clustering spatial sur stops bruts
    _, clustered_stops = cluster_stops_dbscan(
        df,
        raw_stops,
        eps_m=150,
        min_samples=1
    )

    # 4) Regroupement spatio-temporel
    grouped_stops = group_stops_by_time_and_space(
        clustered_stops,
        max_time_gap_s=600,
        max_distance_m=200
    )
    if grouped_stops.empty:
        print(f"Aucun stop agrégé pour {pid}")
        return

    # 5) Classification Home/Work/Autre
    final_stops = classify_home_work(grouped_stops)
    evaluation  = evaluate_home_work_classification(final_stops)

    # 5bis) Calculer les distances entre Home/Work et les "autres"
    def compute_home_work_distances(final_stops):
        from geopy.distance import geodesic

        # Vérifier la présence de Home et Work
        if 'Home' not in final_stops['place_type'].values or 'Work' not in final_stops['place_type'].values:
            return pd.DataFrame(columns=['lat','lon','dist_home_m','dist_work_m','suspect'])

        # Récupérer coordonnées Home et Work
        home = final_stops.loc[final_stops['place_type'] == 'Home', ['lat','lon']].iloc[0]
        work = final_stops.loc[final_stops['place_type'] == 'Work', ['lat','lon']].iloc[0]

        # Filtrer les "autres"
        autres = final_stops[final_stops['place_type'] == 'autre'].copy()

        # Calculer les distances
        autres['dist_home_m'] = autres.apply(
            lambda r: geodesic((r['lat'], r['lon']), (home['lat'], home['lon'])).meters,
            axis=1
        )
        autres['dist_work_m'] = autres.apply(
            lambda r: geodesic((r['lat'], r['lon']), (work['lat'], work['lon'])).meters,
            axis=1
        )

        # Définir un indicateur "suspect" si proche de Home ou Work (< 150 m)
        autres['suspect'] = autres.apply(
            lambda r: 'Oui' if r['dist_home_m'] < 150 or r['dist_work_m'] < 150 else 'Non',
            axis=1
        )

        # Trier par distance à Home
        return autres[['lat','lon','dist_home_m','dist_work_m','suspect']].sort_values('dist_home_m')

    # Appel de la fonction
    autres_with_distances = compute_home_work_distances(final_stops)

    # 6) Vérification (facultative)
    #matched, _ = verify_stop_activities(final_stops, engine, pid)
    #unknowns = matched[matched['place_type']=='autre'] if not matched.empty else pd.DataFrame()

    # 7) Étiquetage des moves
    moves = tag_moves_with_stop_types(moves, final_stops, max_dist_m=200)
    moves['origin_type'     ] = moves.get('origin_type',     'unknown')
    moves['destination_type'] = moves.get('destination_type','unknown')
    moves['transition'] = moves['origin_type'] + ' → ' + moves['destination_type']

    # 7b) Calcul de la distance géographique du move
    moves['dist_m'] = moves.apply(
        lambda r: geodesic(
            (r.lat_origin, r.lon_origin),
            (r.lat_dest,   r.lon_dest)
        ).meters,
        axis=1
    )

    # 7c) Filtrage pour ne garder que les vrais déplacements
    MIN_MOVE_DIST     = 50        # m
    MAX_MOVE_DURATION = 6 * 3600  # s (6 heures)
    moves = moves[
        (moves['dist_m'] >= MIN_MOVE_DIST) &
        (moves['duration_s'] <= MAX_MOVE_DURATION) &
        (moves['origin_type'] != moves['destination_type'])
    ].reset_index(drop=True)

    moves_snapped = snap_moves_to_home_work(moves, final_stops, max_dist_m=150)

    # 8) Sauvegardes CSV
    raw_stops.to_csv(f"data/{pid}_raw_stops.csv", index=False)
    moves    .to_csv(f"data/{pid}_moves_filtered.csv", index=False)

    # 9) Génération du rapport HTML
    section = generate_full_report(
        df_all=df,
        stops_summary_all=raw_stops,
        merged_grouped_stops=grouped_stops,
        final_stops=final_stops,
        final_evaluation_merged=evaluation,
        moves_tagged=moves,
        moves_snapped=moves_snapped,
        pid=pid,
        autres_with_distances=autres_with_distances
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

    print(f"=== Rapport généré → {path_html}===")

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
        df = load_data_and_prepare(engine, pid, max_speed_kmh=150)
        if df.empty:
            print("Aucun point GPS.")
            continue
        generate_report_for_participant(df, pid, engine)

if __name__ == '__main__':
    main()
