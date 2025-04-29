import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

def fig_to_base64(fig):
    """Convertit une figure Matplotlib en base64 pour intégration HTML."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def assign_movement_type(row, speed_thresholds=(1, 5), stop_min_duration=60):
    """Assigne un type de mouvement basé sur la vitesse et la durée d'arrêt."""
    speed = row['speed_kmh']
    duration = row['time_diff_s']

    if pd.isna(speed) or pd.isna(duration):
        return 'unknown'
    if speed < speed_thresholds[0] and duration >= stop_min_duration:
        return 'stop'
    elif speed < speed_thresholds[1]:
        return 'slow_walk'
    elif speed < 15:
        return 'fast_walk'
    elif speed <= 150:
        return 'transport'
    else:
        return 'unknown'

def detect_stops_in_noise(df, speed_threshold=1.0, min_duration_s=60):
    """Détecte les arrêts dans le bruit (cluster -1) basé sur vitesse faible et durée."""
    df_noise = df[(df['cluster_behavior'] == -1) & (df['speed_kmh'] < speed_threshold)].copy()
    df_noise['gap'] = df_noise['timestamp'].diff().dt.total_seconds().fillna(0)
    stop_group = (df_noise['gap'] > min_duration_s).cumsum()
    df_noise['stop_group'] = stop_group

    stops_summary = df_noise.groupby('stop_group').agg(
        start_time=('timestamp', 'min'),
        end_time=('timestamp', 'max'),
        duration_s=('time_diff_s', 'sum'),
        lat=('lat', 'mean'),
        lon=('lon', 'mean'),
        speed_kmh_mean=('speed_kmh', 'mean'),
        n_points=('timestamp', 'count')
    ).reset_index(drop=True)

    stops_summary = stops_summary[stops_summary['duration_s'] >= min_duration_s]
    return stops_summary

def detect_stops_and_generate_figures(df):
    """Détecte les arrêts et génère des figures statistiques pour le rapport final."""
    
    # --- Sécurité : Ajout des colonnes si elles n'existent pas ---
    if 'hour' not in df.columns:
        df['hour'] = df['timestamp'].dt.hour
    if 'weekday' not in df.columns:
        df['weekday'] = df['timestamp'].dt.day_name()
    if 'weekday_num' not in df.columns:
        df['weekday_num'] = df['timestamp'].dt.weekday

    # --- Détecter le type de mouvement ---
    df['movement_type'] = df.apply(assign_movement_type, axis=1)

    figures_base64 = {}

    # --- Figures globales ---
    # 1. Distribution des vitesses
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    df['speed_kmh'].clip(upper=80).hist(bins=60, ax=ax1)
    ax1.set_title("Distribution des vitesses (km/h)")
    figures_base64['distribution_vitesse'] = fig_to_base64(fig1)

    # 2. Vitesses faibles
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    df[df['speed_kmh'] <= 10]['speed_kmh'].hist(bins=30, ax=ax2)
    ax2.set_title("Vitesses ≤ 10 km/h")
    figures_base64['distribution_vitesse_basse'] = fig_to_base64(fig2)

    # 3. Vitesses élevées
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    df[df['speed_kmh'] > 10]['speed_kmh'].hist(bins=30, ax=ax3)
    ax3.set_title("Vitesses > 10 km/h")
    figures_base64['distribution_vitesse_haute'] = fig_to_base64(fig3)

    # 4. Vitesse moyenne par heure
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    df.groupby('hour')['speed_kmh'].mean().plot(kind='bar', ax=ax4)
    ax4.set_title("Vitesse moyenne par heure")
    figures_base64['vitesse_par_heure'] = fig_to_base64(fig4)

    # 5. Vitesse moyenne par jour de la semaine
    fig5, ax5 = plt.subplots(figsize=(8, 4))
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df.groupby('weekday')['speed_kmh'].mean().reindex(weekday_order).plot(kind='bar', ax=ax5)
    ax5.set_title("Vitesse moyenne par jour")
    figures_base64['vitesse_par_jour'] = fig_to_base64(fig5)

    # 6. Types de mouvements
    fig6, ax6 = plt.subplots(figsize=(6, 4))
    df['movement_type'].value_counts().plot(kind='bar', ax=ax6, color=['red', 'orange', 'blue', 'green', 'gray'])
    ax6.set_title("Répartition des types de mouvement")
    figures_base64['types_de_mouvement'] = fig_to_base64(fig6)

    # 7. Clusters détectés
    fig7, ax7 = plt.subplots(figsize=(6, 4))
    df['cluster_behavior'].value_counts().sort_index().plot(kind='bar', ax=ax7)
    ax7.set_title("Répartition des clusters détectés (HDBSCAN)")
    figures_base64['clusters_detectes'] = fig_to_base64(fig7)

    plt.close('all')  # Libérer mémoire matplotlib

    # --- Détection des stops cachés dans le bruit ---
    stops_summary = detect_stops_in_noise(df)

    return df, stops_summary, figures_base64
