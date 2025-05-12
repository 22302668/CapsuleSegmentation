import pandas as pd
from sqlalchemy import text

def verify_stop_activities(stops_summary, engine=None, participant_id="9999965"):
    path = r"C:\Users\22302668\Desktop\CapsuleV2\participant-data-semain43\Activities\Participant9999965-activities.csv"
    df_activities = pd.read_csv(path)

    # # Charger les activités depuis la base
    # with engine.connect() as conn:
    #     df_activities = pd.read_sql_query(text(f"""
    #         SELECT participant_virtual_id, timestamp, activity
    #         FROM detected_activities_record_based_v3
    #         WHERE participant_virtual_id = '{participant_id}'
    #         ORDER BY timestamp ASC
    #     """), con=conn)

    # Convertir timestamps en timezone Europe/Paris
    #df_activities['timestamp'] = pd.to_datetime(df_activities['timestamp'], utc=True).dt.tz_convert('Europe/Paris')
    df_activities['timestamp'] = pd.to_datetime(df_activities['time'], utc=True).dt.tz_convert('Europe/Paris')
    stops_summary['start_time'] = pd.to_datetime(stops_summary['start_time']).dt.tz_localize('Europe/Paris')
    stops_summary['end_time'] = pd.to_datetime(stops_summary['end_time']).dt.tz_localize('Europe/Paris')

    # Trouver l'activité pendant chaque stop
    def get_activity_for_stop(row):
        matches = df_activities[
            (df_activities['timestamp'] >= row['start_time']) &
            (df_activities['timestamp'] <= row['end_time'])
        ]
        if not matches.empty:
            return matches['activity'].iloc[0]  # première activité trouvée
        else:
            return "Unknown"

    # Appliquer à tous les arrêts
    stops_summary['detected_activity'] = stops_summary.apply(get_activity_for_stop, axis=1)

    # Créer un résumé des activités pendant les stops
    summary = (
        stops_summary.groupby('detected_activity')['duration_s']
        .sum()
        .div(60)  # secondes → minutes
        .reset_index()
        .rename(columns={'duration_s': 'durée_totale_min'})
        .sort_values(by='durée_totale_min', ascending=False)
    )

    return stops_summary, summary
