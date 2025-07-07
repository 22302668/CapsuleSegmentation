#!/usr/bin/env python3
import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from pathlib import Path

from load_and_preprocess         import load_data_and_prepare, segment_by_data_weeks
#from movingpandas_stop_detection import detect_stops_with_movingpandas
from scikit_mobility import detect_stops_with_skmob
from classify_home_work          import classify_home_work
from evaluate_home_work          import evaluate_home_work_classification
from detect_stops_and_analyze    import generate_figures
from verify_stop_activities      import verify_stop_activities
from generate_report             import render_segment_report, generate_full_report
from dbscan_clustering           import cluster_stops_dbscan
from split_moves_stops           import split_stops_moves

def generate_report_for_participant(df, participant_id, engine):
    # 1) découpage en plages de jours consécutifs
    segments = segment_by_data_weeks(df)
    if len(segments) <= 1:
        segments = [(df['timestamp'].dt.date.min(), df['timestamp'].dt.date.max())]

    os.makedirs("data", exist_ok=True)
    rapport_path = os.path.join("data", f"{participant_id}_rapport.html")

    all_stops     = []
    segment_htmls = []

    for idx, (d_start, d_end) in enumerate(segments, start=1):
        # 2) extraire le segment
        df_seg = df[
            (df['timestamp'].dt.date >= d_start) &
            (df['timestamp'].dt.date <= d_end)
        ].reset_index(drop=True)

        # 3) détection brute de stops
        # stops_summary = detect_stops_with_movingpandas(
        #     df_seg,
        #     min_duration_minutes=15,
        #     max_diameter_meters=75
        # )
        stops_summary = detect_stops_with_skmob(
            df_seg,
            epsilon_m=75,      # rayon en mètres
            min_time_s=15*60    # durée min en secondes
        )

        if stops_summary.empty:
            segment_htmls.append(
                f"<hr><h2>Segment {idx} : {d_start} → {d_end}</h2>"
                "<p><em>Aucun stop détecté.</em></p>"
            )
            continue

        # 4) clustering spatial DBSCAN
        raw_stops = stops_summary[['start_time','end_time','lat','lon','duration_s']].copy()
        # Note : signature cluster_stops_dbscan(gps_df, stops_df, eps_m, min_samples)
        gps_in_stops, grouped_stops = cluster_stops_dbscan(
            df_seg,
            raw_stops,
            eps_m=150,
            min_samples=1
        )
        if grouped_stops.empty:
            segment_htmls.append(
                f"<hr><h2>Segment {idx} : {d_start} → {d_end}</h2>"
                "<p><em>Aucun stop après DBSCAN.</em></p>"
            )
            continue

        all_stops.append(grouped_stops)

        # 5) classification Home/Work + statistiques + figures
        classified = classify_home_work(grouped_stops)
        evaluation = evaluate_home_work_classification(classified)
        figures    = generate_figures(df_seg, classified, stops_summary)

        # 6) rendu HTML de ce segment
        seg_html = render_segment_report(
            df_seg,
            stops_summary,
            grouped_stops,
            classified,
            evaluation,
            #pd.DataFrame(),  # matched_unknowns_df vide ici
            figures,
            d_start,
            d_end
        )
        segment_htmls.append(seg_html)

    # — fusion finale de tous les stops détectés —
    merged = pd.concat(all_stops, ignore_index=True) if all_stops else pd.DataFrame()
    if not merged.empty:
        # re-classification après fusion
        final_classified = classify_home_work(merged)
        final_evaluation = evaluate_home_work_classification(final_classified)

        # retirer tz pour verify_stop_activities
        final_classified['start_time'] = final_classified['start_time'].dt.tz_localize(None)
        final_classified['end_time']   = final_classified['end_time'].dt.tz_localize(None)

        # matched, _       = verify_stop_activities(
        #     final_classified, engine, participant_id
        # )
        # matched_unknowns = (
        #     matched[matched['place_type']=='autre'].copy()
        #     if not matched.empty else pd.DataFrame()
        # )
    else:
        final_classified = merged
        final_evaluation = {}
        #matched_unknowns = pd.DataFrame()

    # 7) génération de la section « Résultat final »
    # stops_summary_all = detect_stops_with_movingpandas(
    #     df, 
    #     min_duration_minutes=15, 
    #     max_diameter_meters=75
    # )

    stops_summary_all = detect_stops_with_skmob(
        df,
        epsilon_m=75,      # rayon en mètres
        min_time_s=15*60 
    )

    # dans main ou generate_report_for_participant global
    ds1_all, ds2_all = split_stops_moves(df, stops_summary_all)
    ds1_all.to_csv(f"data/{participant_id}_all_stops.csv", index=False)
    ds2_all.to_csv(f"data/{participant_id}_all_moves.csv", index=False)
    
    full_section = generate_full_report(
        df,
        stops_summary_all,       # ← passe les stops bruts MovingPandas
        merged,
        final_classified,
        final_classified,
        final_evaluation,
        #matched_unknowns,
        ds2_all         # ← passe la table de 'autre' jointe aux activités
    )
    # 8) écriture du rapport complet
    with open(rapport_path, "w", encoding="utf-8") as f:
        f.write(
            "<!DOCTYPE html><html><head><meta charset='UTF-8'>"
            "<title>Rapport GPS</title>"
            "<style>"
            "body{font-family:Arial; margin:20px;}"
            "h1,h2,h3{color:#2c3e50;}"
            ".table-container{overflow-x:auto;}"
            "hr{margin:40px 0;}"
            "</style></head><body>"
        )
        f.write(f"<h1>Rapport GPS – Participant {participant_id}</h1>")
        f.write(f"<p><em>Date : {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}</em></p>")
        f.write(full_section)
        for html in segment_htmls:
            f.write(html)
        f.write("</body></html>")
        export_dir = Path("data")
        export_dir.mkdir(parents=True, exist_ok=True)

        final_classified.to_csv(export_dir / f"{participant_id}_classified_places.csv", index=False)
    print(f"Rapport généré → {rapport_path}")


def main():
    load_dotenv()
    url = (
        f"postgresql+psycopg2://{os.getenv('PG_USER')}:" 
        f"{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:" 
        f"{os.getenv('PG_PORT')}/{os.getenv('PG_DB')}"
    )
    engine = create_engine(url)

    with engine.connect() as conn:
        participants = pd.read_sql_query(
            text("SELECT DISTINCT participant_virtual_id FROM gps_all_participants"),
            conn
        )['participant_virtual_id'].tolist()

    for pid in participants:
        print(f"\n=== Participant {pid} ===")
        df = load_data_and_prepare(engine, pid)
        if df.empty:
            print("Aucun point GPS.")
            continue
        generate_report_for_participant(df, pid, engine)


if __name__ == "__main__":
    main()
