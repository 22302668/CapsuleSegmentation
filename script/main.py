from load_and_preprocess            import load_data_and_prepare, segment_by_data_weeks
from generate_report                import render_segment_report, generate_full_report
from movingpandas_stop_detection    import detect_stops_with_movingpandas
from detect_stops_and_analyze       import generate_figures
from verify_stop_activities         import verify_stop_activities
from group_stops                    import merge_stops
from classify_home_work             import classify_home_work
from evaluate_home_work import evaluate_home_work_classification
from merge_close_stops              import merge_close_stops

import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv


def generate_report_for_participant(df, participant_id, engine):
    segments = segment_by_data_weeks(df)
    is_single_segment = len(segments) <= 1

    if is_single_segment:
        print("Un seul segment détecté : traitement global sans découpage.")
        segments = [(df['timestamp'].dt.date.min(), df['timestamp'].dt.date.max())]

    os.makedirs("data", exist_ok=True)
    rapport_filename = f"{participant_id}_rapport.html"
    rapport_path     = os.path.join("data", rapport_filename)

    all_grouped_stops = []
    segment_htmls     = []

    if not is_single_segment:
        print("Segments de collecte détectés :")
        for i, (d_start, d_end) in enumerate(segments, start=1):
            print(f"  Segment {i} : {d_start} → {d_end}")

        for idx, (d_start, d_end) in enumerate(segments, start=1):
            print(f"\n--- Traitement du segment {idx} : {d_start} → {d_end} ---")
            mask   = (df['timestamp'].dt.date >= d_start) & (df['timestamp'].dt.date <= d_end)
            df_seg = df.loc[mask].reset_index(drop=True)
            print(f"Points GPS dans ce segment : {len(df_seg)}")

            print("Détection des stops avec MovingPandas…")
            stops_summary = detect_stops_with_movingpandas(df_seg, min_duration_minutes=6, max_diameter_meters=30)
            grouped_stops = merge_stops(stops_summary)

            if not grouped_stops.empty and 'start_time' in grouped_stops.columns:
                all_grouped_stops.append(grouped_stops)
            else:
                html_note = f"""
                <hr style="margin:40px 0;">
                <h2 id="segment_{d_start}_{d_end}">Segment {idx} : {d_start} → {d_end}</h2>
                <p><em>Aucun stop détecté sur ce segment.</em></p>
                """
                segment_htmls.append(html_note)
                continue

            classified_stops = classify_home_work(grouped_stops)
            final_stops = merge_close_stops(classified_stops, max_distance_m=150)
            evaluation = evaluate_home_work_classification(final_stops)
            figures_base64 = generate_figures(df_seg, classified_stops, stops_summary)

            segment_html = render_segment_report(
                df_seg, stops_summary, grouped_stops, final_stops, evaluation,
                pd.DataFrame(), figures_base64, d_start, d_end
            )
            segment_htmls.append(segment_html)

    # 🔁 Partie fusion finale (toujours faite, même si 1 seul segment)
    print("\nFusion finale et génération de la section « Résultat final »…")
    if is_single_segment:
        d_start, d_end = segments[0]
        df_seg = df.copy()
        stops_summary = detect_stops_with_movingpandas(df_seg, min_duration_minutes=6, max_diameter_meters=30)
        grouped_stops = merge_stops(stops_summary)
        all_grouped_stops.append(grouped_stops)

    merged_grouped_stops   = pd.concat(all_grouped_stops, ignore_index=True)
    final_classified_stops = classify_home_work(merged_grouped_stops)
    final_merged_stops     = merge_close_stops(final_classified_stops, max_distance_m=150)
    final_evaluation_merged = evaluate_home_work_classification(final_merged_stops)

    stops_with_activities, _ = verify_stop_activities(final_classified_stops, engine, participant_id)
    matched_unknowns_df      = stops_with_activities[stops_with_activities['place_type'] == 'autre'].copy()

    final_html_section = generate_full_report(
        df, merged_grouped_stops, final_merged_stops, final_evaluation_merged, matched_unknowns_df
    )

    with open(rapport_path, "w", encoding="utf-8") as f:
        f.write("""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Rapport GPS – MovingPandas</title>
<style>body { font-family: Arial, sans-serif; margin: 20px; } h1, h2, h3, h4 { color: #2c3e50; } .table-container { overflow-x: auto; } hr { margin: 40px 0; }</style>
</head><body>
""")
        f.write(f"<h1>Rapport GPS – Participant {participant_id}</h1>\n")
        f.write(f"<p><em>Date de génération : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>\n")
        f.write(final_html_section)

    with open(rapport_path, "a", encoding="utf-8") as f:
        for html_seg in segment_htmls:
            f.write(html_seg)
        f.write("\n</body>\n</html>")

    print(f"\nRapport généré pour le participant : {rapport_path}")

def main():
    load_dotenv()
    url = (
        f"postgresql+psycopg2://{os.getenv('PG_USER')}:" 
        f"{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:" 
        f"{os.getenv('PG_PORT')}/{os.getenv('PG_DB')}"
    )
    engine = create_engine(url)

    with engine.connect() as conn:
        participants = pd.read_sql_query(text("""
            SELECT DISTINCT participant_virtual_id
            FROM gps_mesures
        """), conn)['participant_virtual_id'].tolist()

    for participant_id in participants:
        print(f"\n=== Traitement du participant : {participant_id} ===")
        try:
            df = load_data_and_prepare(engine, participant_id)
            if df.empty:
                print("→ Données GPS absentes pour ce participant.")
                continue
            generate_report_for_participant(df, participant_id, engine)
        except Exception as e:
            print(f"❌ Erreur pour le participant {participant_id} : {e}")


if __name__ == "__main__":
    main()
