import pandas as pd
from sqlalchemy import text

def verify_stop_activities(final_stops, engine, participant_id):
    # Charger les activités depuis la table unifiée
    with engine.connect() as conn:
        df_activities = pd.read_sql_query(text("""
            SELECT
              participant_id,
              "Date de début" AS timestamp,
              "Position" AS activity
            FROM public.activity_all_participants
            WHERE participant_id = :participant_id
            ORDER BY "Date de début" ASC
        """), con=conn, params={"participant_id": participant_id})

    # Conversion des dates
    df_activities['timestamp'] = pd.to_datetime(df_activities['timestamp'], utc=True).dt.tz_convert('Europe/Paris')

    final_stops['start_time'] = pd.to_datetime(final_stops['start_time'])
    # Si la série n'est pas encore tz-aware, on localise, sinon on convertit
    if final_stops['start_time'].dt.tz is None:
        final_stops['start_time'] = final_stops['start_time'].dt.tz_localize('Europe/Paris')
    else:
        final_stops['start_time'] = final_stops['start_time'].dt.tz_convert('Europe/Paris')

    final_stops['end_time']   = pd.to_datetime(final_stops['end_time'])
    if final_stops['end_time'].dt.tz is None:
        final_stops['end_time'] = final_stops['end_time'].dt.tz_localize('Europe/Paris')
    else:
        final_stops['end_time'] = final_stops['end_time'].dt.tz_convert('Europe/Paris')

    # Trouver l'activité pendant chaque stop
    def get_activity_for_stop(row):
        matches = df_activities[
            (df_activities['timestamp'] >= row['start_time']) &
            (df_activities['timestamp'] <= row['end_time'])
        ]
        return matches['activity'].iloc[0] if not matches.empty else "autre"

    final_stops['matched_activity'] = final_stops.apply(get_activity_for_stop, axis=1)

    # Résumé des durées par activité
    summary = (
        final_stops.groupby('matched_activity')['duration_s']
        .sum()
        .div(60)
        .reset_index()
        .rename(columns={'duration_s': 'durée_totale_min'})
        .sort_values(by='durée_totale_min', ascending=False)
    )

    return final_stops, summary
