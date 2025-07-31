import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import numpy as np
from matplotlib.patches import Patch

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

def plot_daily_hourly_speed_patterns(df):
    df = df[df['timestamp'].notna()].copy()
    df['hour'] = df['timestamp'].dt.hour
    df['date'] = df['timestamp'].dt.date
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'] >= 5

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Partie semaine
    for date, group in df[~df['is_weekend']].groupby('date'):
        hourly = group.groupby('hour')['speed_kmh'].mean()
        if not hourly.empty:
            ax1.plot(hourly.index, hourly, label=str(date))
    ax1.set_title("Vitesse moyenne par heure – Semaine")
    ax1.set_ylabel("Vitesse (km/h)")
    ax1.legend(loc='upper right', fontsize='x-small')

    # Partie weekend
    for date, group in df[df['is_weekend']].groupby('date'):
        hourly = group.groupby('hour')['speed_kmh'].mean()
        if not hourly.empty:
            ax2.plot(hourly.index, hourly, label=str(date))
    ax2.set_title("Vitesse moyenne par heure – Weekend")
    ax2.set_xlabel("Heure")
    ax2.set_ylabel("Vitesse (km/h)")
    ax2.legend(loc='upper right', fontsize='x-small')

    fig.tight_layout()
    return fig_to_base64(fig)

def plot_heatmap_date_hour(df):
    # Préparation
    df = df[df['timestamp'].notna()].copy()
    df['hour'] = df['timestamp'].dt.hour
    df['date'] = df['timestamp'].dt.date

    # Grouper : nombre de points par date et heure
    grouped = df.groupby(['date', 'hour']).size().unstack(fill_value=0)

    # Trier les dates (si pas déjà)
    grouped = grouped.sort_index()

    # Création de la heatmap
    fig, ax = plt.subplots(figsize=(14, max(6, len(grouped) * 0.35)))  # hauteur dynamique selon nbre de jours
    sns.heatmap(grouped, cmap="Blues", annot=True, fmt='d', linewidths=.5, ax=ax)

    ax.set_title("Heatmap des fréquences GPS – Date x Heure")
    ax.set_xlabel("Heure")
    ax.set_ylabel("Date")
    plt.tight_layout()

    return fig_to_base64(fig)

def plot_heatmap_vitesse_date_hour(df):
    df = df[df['timestamp'].notna() & df['speed_kmh'].notna()].copy()
    df['hour'] = df['timestamp'].dt.hour
    df['date'] = df['timestamp'].dt.date

    # Grouper par date et heure → moyenne des vitesses
    grouped = df.groupby(['date', 'hour'])['speed_kmh'].mean().unstack(fill_value=0)

    # Trier les dates
    grouped = grouped.sort_index()

    # Créer la heatmap
    fig, ax = plt.subplots(figsize=(14, max(6, len(grouped) * 0.35)))
    sns.heatmap(grouped, cmap="YlOrRd", annot=True, fmt='.1f', linewidths=.5, ax=ax)

    ax.set_title("Heatmap des vitesses moyennes – Date x Heure")
    ax.set_xlabel("Heure")
    ax.set_ylabel("Date")
    plt.tight_layout()

    return fig_to_base64(fig)

def plot_combined_confidence_score(classified_stops: pd.DataFrame) -> str:
    """
    Génère un graphique du score de confiance combiné (durée + fréquence) pour les lieux Home / Work.
    """
    df = classified_stops[classified_stops['place_type'].isin(['Home', 'Work'])].copy()
    if df.empty:
        return ""

    # Calcul du score temps
    max_duration = df['duration_s'].max()
    df['score_duree'] = df['duration_s'] / max_duration

    # Calcul du score de fréquence (nb de jours distincts)
    df['merged_days'] = df['merged_starts'].apply(lambda lst: set(pd.to_datetime(lst).date) if isinstance(lst, list) else set())
    df['nb_days'] = df['merged_days'].apply(len)
    max_days = df['nb_days'].max()
    df['score_frequence'] = df['nb_days'] / max_days

    # Score combiné pondéré (50% durée, 50% fréquence)
    df['score_combine'] = ((df['score_duree'] + df['score_frequence']) / 2 * 100).round(1)

    df['label'] = df.apply(lambda row: f"{row['place_type']} ({round(row['lat'], 3)}, {round(row['lon'], 3)})", axis=1)

    # Plot
    plt.figure(figsize=(10, max(4, 0.5 * len(df))))
    palette = {'Home': '#3498db', 'Work': '#9b59b6'}

    ax = sns.barplot(
        data=df,
        x="score_combine",
        y="label",
        hue="place_type",
        dodge=False,
        palette=palette
    )

    for p in ax.patches:
        width = p.get_width()
        ax.annotate(f"{width:.0f}%",
                    (width + 1, p.get_y() + p.get_height() / 2),
                    ha='left', va='center', fontsize=9, color='black')

    plt.title("Score combiné Home / Work (durée + fréquence)", fontsize=14)
    plt.xlabel("Score combiné (%)")
    plt.ylabel("Lieu (type + coordonnées)")
    plt.xlim(0, 110)
    plt.legend(title="Type de lieu", loc="lower right")
    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    base64_img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return base64_img

def generate_figures(df, classified_stops, stops_summary=None):
    figures_base64 = {}

    df = enrich_time_columns(df)

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    df['speed_kmh'].clip(upper=80).hist(bins=60, ax=ax1)
    ax1.set_title("Distribution des vitesses (km/h)")
    figures_base64['distribution_vitesse'] = fig_to_base64(fig1)

    # —————————————————————————————————————————
    # Histogrammes segmentés pour 0–3.5, 3.5–6.5 et 6.5–10 km/h
    # —————————————————————————————————————————

    # 0–3.5 km/h
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    mask1 = (df['speed_kmh'] > 0) & (df['speed_kmh'] <= 3.5)
    df.loc[mask1, 'speed_kmh'].hist(bins=np.linspace(0, 3.5, 15), ax=ax2)
    ax2.set_title("Distribution des vitesses lentes (0–3,5 km/h)")
    ax2.set_xlabel("Vitesse (km/h)")
    ax2.set_ylabel("Nombre de points")
    figures_base64['dist_vitesse_0_3_5'] = fig_to_base64(fig2)

    # 3.5–6.5 km/h
    fig6, ax6 = plt.subplots(figsize=(8, 4))
    mask2 = (df['speed_kmh'] > 3.5) & (df['speed_kmh'] <= 6.5)
    df.loc[mask2, 'speed_kmh'].hist(bins=np.linspace(3.5, 6.5, 15), ax=ax6)
    ax6.set_title("Distribution des vitesses de marche (3,5–6,5 km/h)")
    ax6.set_xlabel("Vitesse (km/h)")
    ax6.set_ylabel("Nombre de points")
    figures_base64['dist_vitesse_3_5_6_5'] = fig_to_base64(fig6)

    # 6.5–10 km/h
    fig7, ax7 = plt.subplots(figsize=(8, 4))
    mask3 = (df['speed_kmh'] > 6.5) & (df['speed_kmh'] <= 10)
    df.loc[mask3, 'speed_kmh'].hist(bins=np.linspace(6.5, 10, 15), ax=ax7)
    ax7.set_title("Distribution des vitesses rapides (6,5–10 km/h)")
    ax7.set_xlabel("Vitesse (km/h)")
    ax7.set_ylabel("Nombre de points")
    figures_base64['dist_vitesse_6_5_10'] = fig_to_base64(fig7)


    fig3, ax3 = plt.subplots(figsize=(8, 4))
    df[df['speed_kmh'] > 10]['speed_kmh'].hist(bins=30, ax=ax3)
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

    figures_base64['vitesse_hebdo_horaire'] = plot_daily_hourly_speed_patterns(df)

    figures_base64['heatmap_date_hour'] = plot_heatmap_date_hour(df)
    figures_base64['heatmap_vitesse_date_hour'] = plot_heatmap_vitesse_date_hour(df)

    if classified_stops is not None:
        figures_base64['confidence_score_home_work'] = plot_combined_confidence_score(classified_stops)

    plt.close('all')
    return figures_base64
