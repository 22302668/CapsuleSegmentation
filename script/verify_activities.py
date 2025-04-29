import pandas as pd
from sqlalchemy import text


def verify_activities(df, engine, participant_id='9999932'):
    """
    Attribue l'activité théorique à chaque point GPS en fonction de la base de données,
    et génère un tableau croisant Cluster HDBSCAN vs Activité détectée.

    Args:
        df (pd.DataFrame): Données GPS traitées.
        engine (sqlalchemy.Engine): Connexion SQLAlchemy à la base de données.
        participant_id (str): ID du participant à utiliser.

    Returns:
        df (pd.DataFrame): DataFrame enrichi avec colonne 'activity_detected'.
        cross_tab (pd.DataFrame): Tableau croisant cluster_label vs activity_detected.
    """

    # Charger les activités détectées depuis la base
    with engine.connect() as conn:
        df_activities = pd.read_sql_query(text(f"""
            SELECT participant_virtual_id, timestamp, activity
            FROM detected_activities_record_based_v3
            WHERE participant_virtual_id = '{participant_id}'
            ORDER BY timestamp ASC
        """), con=conn)

    # S'assurer que les dates sont au bon format
    df_activities['timestamp'] = pd.to_datetime(df_activities['timestamp'], utc=True).dt.tz_convert('Europe/Paris')

    # Attribuer à chaque point GPS l'activité la plus récente
    df = df.sort_values('timestamp')
    df_activities = df_activities.rename(columns={'timestamp': 'activity_timestamp'})

    df['activity_detected'] = pd.merge_asof(
        df,
        df_activities.sort_values('activity_timestamp'),
        left_on='timestamp',
        right_on='activity_timestamp',
        direction='backward'
    )['activity']

    # Remplacer les activités manquantes par "Unknown"
    df['activity_detected'] = df['activity_detected'].fillna('Unknown')

    # Générer le tableau croisant cluster vs activité
    cross_tab = pd.crosstab(df['cluster_label'], df['activity_detected'])

    return df, cross_tab
