import pandas as pd
from sqlalchemy import text

def verify_activities(df, engine=None, participant_id='999996'):
    """
    Attribue l'activité théorique à chaque point GPS à partir de la BDD,
    et génère un tableau croisant movement_type vs activité détectée (MovingPandas).
    """
    path = r"C:\Users\22302668\Desktop\CapsuleV2\participant-data-semain43\Activities\Participant9999965-activities.csv"
    #activities = pd.read_csv(path)
    df_activities = pd.read_csv(path)
    # # Charger les activités depuis la base
    # with engine.connect() as conn:
    #     df_activities = pd.read_sql_query(text(f"""
    #         SELECT participant_virtual_id, timestamp, activity
    #         FROM detected_activities_record_based_v3
    #         WHERE participant_virtual_id = '{participant_id}'
    #         ORDER BY timestamp ASC
    #     """), con=conn)

    # Formatage des timestamps
    #df_activities['timestamp'] = pd.to_datetime(df_activities['timestamp'], utc=True).dt.tz_convert('Europe/Paris')
    df_activities['timestamp'] = pd.to_datetime(df_activities['time'], utc=True).dt.tz_convert('Europe/Paris')

    df = df.sort_values('timestamp')
    df_activities = df_activities.rename(columns={'timestamp': 'activity_timestamp'})

    # Merge asof pour récupérer l'activité correspondante à chaque point GPS
    df['activity_detected'] = pd.merge_asof(
        df,
        df_activities.sort_values('activity_timestamp'),
        left_on='timestamp',
        right_on='activity_timestamp',
        direction='backward'
    )['activity'].fillna('Unknown')

    # Générer le tableau croisé
    if 'movement_type' in df.columns:
        cross_tab = pd.crosstab(df['movement_type'], df['activity_detected'])
    else:
        cross_tab = None

    return df, cross_tab
