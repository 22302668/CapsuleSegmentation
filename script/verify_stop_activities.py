import pandas as pd
from sqlalchemy import text

def verify_stop_activities(final_stops, engine, participant_id):
    # path = r"C:\Users\22302668\Desktop\CapsuleV2\participant-data-semain43\Activities\Participant9999965-activities.csv"
    # df_activities = pd.read_csv(path)

    # Charger les activités depuis la base
    with engine.connect() as conn:
        df_activities = pd.read_sql_query(text(f"""
                SELECT
                  p.participant_virtual_id,
                  t."timestamp",
                  t.activity
                FROM public."tabletActivityApp" AS t
                JOIN public.kit AS k
                  ON t.tablet_id = k.tablet_id
                JOIN public."campaignParticipantKit" AS cp
                  ON k.id = cp.kit_id
                JOIN public.participant AS p
                  ON cp.participant_id = p.id
                WHERE
                  p.participant_virtual_id = '{participant_id}'
                  AND t."timestamp" BETWEEN cp.start_date AND cp.end_date
                ORDER BY t."timestamp" ASC
            """),
            con=conn)

    df_activities['timestamp'] = pd.to_datetime(df_activities['timestamp'], utc=True).dt.tz_convert('Europe/Paris')
    # df_activities['timestamp'] = pd.to_datetime(df_activities['time'], utc=True).dt.tz_convert('Europe/Paris')
    final_stops['start_time'] = pd.to_datetime(final_stops['start_time']).dt.tz_localize('Europe/Paris')
    final_stops['end_time'] = pd.to_datetime(final_stops['end_time']).dt.tz_localize('Europe/Paris')

    # Trouver l'activité pendant chaque stop
    def get_activity_for_stop(row):
        matches = df_activities[
            (df_activities['timestamp'] >= row['start_time']) &
            (df_activities['timestamp'] <= row['end_time'])
        ]
        if not matches.empty:
            return matches['activity'].iloc[0]  # première activité trouvée
        else:
            return "autre"

    # Appliquer à tous les arrêts
    final_stops['matched_activity'] = final_stops.apply(get_activity_for_stop, axis=1)

    # Créer un résumé des activités pendant les stops
    summary = (
        final_stops.groupby('matched_activity')['duration_s']
        .sum()
        .div(60)  # secondes → minutes
        .reset_index()
        .rename(columns={'duration_s': 'durée_totale_min'})
        .sort_values(by='durée_totale_min', ascending=False)
    )

    return final_stops, summary
