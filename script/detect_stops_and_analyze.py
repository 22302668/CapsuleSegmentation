# Version nettoyée et optimisée du script de génération de rapport GPS avec MovingPandas

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def enrich_time_columns(df):
    time_col = 'start_time' if 'start_time' in df.columns else 'timestamp'
    df['hour'] = df[time_col].dt.hour
    df['weekday'] = df[time_col].dt.day_name()
    df['weekday_num'] = df[time_col].dt.weekday
    df['date'] = df[time_col].dt.date
    df['dayofweek'] = df[time_col].dt.dayofweek
    df['is_weekend'] = df['dayofweek'] >= 5
    return df

def assign_movement_type(row, speed_thresholds=(1, 5), stop_min_duration=60):
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

def plot_daily_hourly_stop_patterns(stops_summary):
    stops_summary = stops_summary[stops_summary['start_time'].notna()].copy()
    enrich_time_columns(stops_summary)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for date, group in stops_summary[~stops_summary['is_weekend']].groupby('date'):
        hourly = group.groupby('hour')['duration_s'].sum() / 60
        if hourly.sum() > 10:
            ax1.plot(hourly.index, hourly.clip(upper=1000), label=str(date))
    ax1.set_title("Semaine")
    ax1.set_ylabel("Durée des stops (min)")
    ax1.legend()

    for date, group in stops_summary[stops_summary['is_weekend']].groupby('date'):
        hourly = group.groupby('hour')['duration_s'].sum() / 60
        if hourly.sum() > 10:
            ax2.plot(hourly.index, hourly.clip(upper=1000), label=str(date))
    ax2.set_title("Weekend")
    ax2.set_ylabel("Durée des stops (min)")
    ax2.set_xlabel("Heure")
    ax2.legend()

    fig.tight_layout()
    return fig_to_base64(fig)

def generate_figures(df, stops_summary=None):
    figures_base64 = {}

    df = enrich_time_columns(df)
    df['movement_type'] = df.apply(assign_movement_type, axis=1)

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    df['speed_kmh'].clip(upper=80).hist(bins=60, ax=ax1)
    ax1.set_title("Distribution des vitesses (km/h)")
    figures_base64['distribution_vitesse'] = fig_to_base64(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    df[df['speed_kmh'] <= 10]['speed_kmh'].hist(bins=30, ax=ax2)
    ax2.set_title("Vitesses ≤ 10 km/h")
    figures_base64['distribution_vitesse_basse'] = fig_to_base64(fig2)

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    df[df['speed_kmh'] > 10]['speed_kmh'].hist(bins=30, ax=ax3)
    ax3.set_title("Vitesses > 10 km/h")
    figures_base64['distribution_vitesse_haute'] = fig_to_base64(fig3)

    fig4, ax4 = plt.subplots(figsize=(8, 4))
    df.groupby('hour')['speed_kmh'].mean().plot(kind='bar', ax=ax4)
    ax4.set_title("Vitesse moyenne par heure")
    figures_base64['vitesse_par_heure'] = fig_to_base64(fig4)

    fig5, ax5 = plt.subplots(figsize=(8, 4))
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df.groupby('weekday')['speed_kmh'].mean().reindex(weekday_order).plot(kind='bar', ax=ax5)
    ax5.set_title("Vitesse moyenne par jour")
    figures_base64['vitesse_par_jour'] = fig_to_base64(fig5)

    fig6, ax6 = plt.subplots(figsize=(6, 4))
    df['movement_type'].value_counts().plot(kind='bar', ax=ax6, color=['red', 'orange', 'blue', 'green', 'gray'])
    ax6.set_title("Répartition des types de mouvement")
    figures_base64['types_de_mouvement'] = fig_to_base64(fig6)

    if stops_summary is not None and not stops_summary.empty:
        stops_summary = enrich_time_columns(stops_summary)

        fig7, ax7 = plt.subplots(figsize=(6, 5))
        sc = ax7.scatter(stops_summary['lon'], stops_summary['lat'], c=stops_summary['duration_s'], cmap='Reds', s=50, alpha=0.7)
        plt.colorbar(sc, ax=ax7, label='Durée (s)')
        ax7.set_title("Positions des stops (durée en couleur)")
        figures_base64['stops_dispersion'] = fig_to_base64(fig7)

        fig8, ax8 = plt.subplots(figsize=(6, 4))
        stops_summary['duration_s'].hist(bins=20, ax=ax8)
        ax8.set_title("Durée des stops (en secondes)")
        figures_base64['stops_duree'] = fig_to_base64(fig8)

        fig9, ax9 = plt.subplots(figsize=(6, 4))
        stops_summary['hour'].value_counts().sort_index().plot(kind='bar', ax=ax9)
        ax9.set_title("Nombre de stops par heure")
        figures_base64['stops_par_heure'] = fig_to_base64(fig9)

        figures_base64['stop_weekday_vs_weekend'] = plot_daily_hourly_stop_patterns(stops_summary)

    figures_base64.update(generate_frequency_analysis(df))
    plt.close('all')
    return figures_base64

def generate_frequency_analysis(df):
    figures = {}

    df = enrich_time_columns(df)

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    df.groupby('date').size().plot(kind='bar', ax=ax1)
    ax1.set_title("Nombre de points GPS par jour")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Nombre de points")
    figures['points_par_jour'] = fig_to_base64(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    df.groupby('hour').size().plot(kind='bar', ax=ax2)
    ax2.set_title("Nombre de points GPS par heure")
    ax2.set_xlabel("Heure")
    ax2.set_ylabel("Nombre de points")
    figures['points_par_heure'] = fig_to_base64(fig2)

    pivot = df.pivot_table(index='weekday_num', columns='hour', values='timestamp', aggfunc='count', fill_value=0)
    weekday_map = {0: 'Lundi', 1: 'Mardi', 2: 'Mercredi', 3: 'Jeudi', 4: 'Vendredi', 5: 'Samedi', 6: 'Dimanche'}
    pivot.index = pivot.index.map(weekday_map)
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    sns.heatmap(pivot, cmap="Blues", linewidths=0.5, annot=True, fmt="d", ax=ax3)
    ax3.set_title("Heatmap fréquence GPS (jour x heure)")
    ax3.set_xlabel("Heure")
    ax3.set_ylabel("Jour de la semaine")
    figures['heatmap_frequence'] = fig_to_base64(fig3)

    plt.close('all')
    return figures
