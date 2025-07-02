import seaborn as sns
import pandas as pd
from io import BytesIO
import base64
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import matplotlib.dates as mdates

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def evaluate_home_work_classification(classified_stops):
    """
    Retourne un dictionnaire contenant :
     1. Le nombre de lieux détectés par type ('Home', 'Work', 'autre')
     2. La fréquence de visites (nombre d'intervalles) par type
     3. Un graphique (base64) de la répartition horaire des passages
     4. La durée cumulée (en minutes) passée sur chaque type

    Cette fonction est maintenant robuste si 'merged_starts' est absent :
    - Dans ce cas, on crée une colonne temporaire merged_starts où chaque
      stop ne contient qu'un unique intervalle (son start_time).
    """

    df = classified_stops.copy()

    # Si la colonne 'merged_starts' n’existe pas, on la crée.
    # Chaque lieu aura alors une liste contenant son propre start_time string.
    if 'merged_starts' not in df.columns:
        # On passe par dt.strftime pour obtenir une chaîne de caractères similaire
        df['merged_starts'] = df['start_time'].dt.strftime('%Y-%m-%d %H:%M:%S').apply(lambda s: [s])

    results = {}

    # 1. Nombre de lieux détectés par type
    type_counts = df['place_type'].value_counts()
    results['nombre_lieux_par_type'] = type_counts.to_dict()

    # 2. Fréquence de visite : on compte le nombre d’intervalles par lieu
    df['nb_intervalles'] = df['merged_starts'].apply(lambda x: len(x) if isinstance(x, list) else 1)
    freq_summary = df.groupby('place_type')['nb_intervalles'].describe()
    results['frequence_par_type'] = freq_summary.to_dict()

    # 3. Horaires typiques : extraire l’heure de chaque timestamp dans merged_starts
    time_stats = []
    for _, row in df.iterrows():
        starts = row['merged_starts']
        if isinstance(starts, list):
            for ts in starts:
                h = pd.to_datetime(ts).hour
                time_stats.append({
                    'place_type': row['place_type'],
                    'hour': h
                })

    df_time = pd.DataFrame(time_stats)
    if not df_time.empty:
        plt.figure(figsize=(10, 4))
        sns.histplot(
            data=df_time,
            x='hour',
            hue='place_type',
            multiple='stack',
            bins=24
        )
        plt.title("Répartition des heures de passage par type de lieu")
        plt.xlabel("Heure (0–23 h)")
        plt.ylabel("Nombre d'intervalles")
        plt.tight_layout()
        results['graph_hourly_distribution'] = fig_to_base64(plt.gcf())
        plt.close()

    # 4. Durée cumulée (en minutes) par type
    duration_stats = df.groupby('place_type')['duration_s'].sum() / 60.0
    results['duree_cumulee_minutes'] = duration_stats.to_dict()

    return results

def plot_rolling_speed(df, window_min=10) -> str:
    """
    Trace la vitesse moyenne glissante minute-par-minute,
    avec une couleur différente par date.
    
    Args:
        df (pd.DataFrame): doit contenir ['timestamp','speed_kmh'].
        window_min (int): taille de la fenêtre glissante (en minutes).
    Returns:
        str: HTML embed Plotly
    """
    # 1) Préparation et resampling à 1 min
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    
    # On interpole les vitesses manquantes puis on lisse
    speed_1min = df['speed_kmh'].resample('1T').mean().interpolate()
    speed_smooth = speed_1min.rolling(window=window_min, min_periods=1).mean()
    
    # 2) Remettre en DataFrame et extraire la date
    dfm = speed_smooth.reset_index().rename(columns={'speed_kmh':'speed_kmh_smooth'})
    dfm['date'] = dfm['timestamp'].dt.date
    
    # 3) Tracé Plotly
    fig = px.line(
        dfm,
        x='timestamp',
        y='speed_kmh_smooth',
        color='date',
        labels={
            'timestamp': "Heure",
            'speed_kmh_smooth': f"Vitesse lissée sur {window_min} min (km/h)",
            'date': "Date"
        },
        title=f"Vitesse moyenne glissante ({window_min} min) par minute et par jour"
    )
    fig.update_xaxes(
        dtick=600000,  # tick toutes les 10 min
        tickformat="%H:%M"
    )
    fig.update_layout(
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(title="Date", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def plot_rolling_speed_with_place(df_all, stops_df, window_min=10):
    # 1) Calcul de la vitesse lissée au pas d'1 minute
    df = df_all.copy()
    df = df.set_index('timestamp').resample('1T').mean().interpolate()
    df['speed_smooth'] = df['speed_kmh'].rolling(window=window_min, min_periods=1).mean()
    df = df.reset_index()

    # 2) On crée une colonne place_type par intervalle
    #    Par défaut 'autre'
    df['place_type'] = 'autre'
    for _, stop in stops_df.iterrows():
        mask = (df['timestamp'] >= stop['start_time']) & (df['timestamp'] <= stop['end_time'])
        df.loc[mask, 'place_type'] = stop['place_type']

    # 3) Tracé avec Plotly Express
    fig = px.line(
        df,
        x='timestamp',
        y='speed_smooth',
        color='place_type',
        line_dash='place_type',
        labels={
            'timestamp': "Temps",
            'speed_smooth': f"Vitesse lissée sur {window_min} min (km/h)",
            'place_type': "Type de lieu"
        },
        title=f"Vitesse lissée sur {window_min} min, coloriée par place_type"
    )
    fig.update_layout(legend=dict(title="Lieu"))
    return fig.to_html(full_html=False, include_plotlyjs='cdn')
