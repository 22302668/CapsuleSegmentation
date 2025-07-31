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
from plotly.subplots import make_subplots

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
        df['merged_starts'] = df['start_time'].dt.tz_convert('Europe/Paris').dt.strftime('%Y-%m-%d %H:%M:%S').apply(lambda s: [s])

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
                h = pd.to_datetime(ts, utc=True).tz_convert('Europe/Paris').hour
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
        fig = plt.gcf()
        results['graph_hourly_distribution'] = fig_to_base64(fig)
        plt.close(fig)

    # 4. Durée cumulée (en minutes) par type
    duration_stats = df.groupby('place_type')['duration_s'].sum() / 60.0
    results['duree_cumulee_minutes'] = duration_stats.to_dict()

    return results

def plot_rolling_speed(df, stops, moves, window_min=10) -> str:
    """
    Deux sous-graphiques :
    - Haut : vitesse brute + lissée colorée par type de lieu.
    - Bas : timeline des déplacements (moves) avec transitions.
    """
    # --- Préparation des données ---
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert("Europe/Paris")
    df = df.set_index('timestamp').sort_index()
    speed_1min = df['speed_kmh'].resample('1T').mean().interpolate()
    speed_smooth = speed_1min.rolling(window=window_min, min_periods=1, center=True).mean()
    dfm = speed_smooth.reset_index().rename(columns={'speed_kmh':'speed_kmh_smooth'})
    
    intervals = pd.IntervalIndex.from_arrays(stops['start_time'], stops['end_time'], closed='both')
    s_place = pd.Series(stops['place_type'].values, index=intervals)
    dfm['place_type'] = dfm['timestamp'].apply(
        lambda ts: s_place[s_place.index.contains(ts)].iloc[0] if not s_place[s_place.index.contains(ts)].empty else 'Move'
    )

    moves['start_time'] = pd.to_datetime(moves['start_time'], utc=True).dt.tz_convert("Europe/Paris")
    moves['end_time'] = pd.to_datetime(moves['end_time'], utc=True).dt.tz_convert("Europe/Paris")
    total_dist = moves['dist_m'].sum()
    total_moves = len(moves)
    total_duration = moves['duration_s'].sum() / 60

    # --- Création des subplots ---
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, 
        row_heights=[0.7, 0.3], vertical_spacing=0.05,
        subplot_titles=(f"Vitesse (ROLLING {window_min} min)", "Déplacements (Timeline)")
    )

    # --- Subplot 1 : Vitesse ---
    # Vitesse brute
    fig.add_trace(go.Scatter(
        x=speed_1min.index, y=speed_1min,
        mode='lines', name='Vitesse brute',
        line=dict(color='lightgray', width=1, dash='dot')
    ), row=1, col=1)

    # Vitesse lissée par type
    color_map = {'Home': 'blue', 'Work': 'green', 'autre': 'gray', 'Move': 'orange'}
    for place, color in color_map.items():
        sub = dfm[dfm['place_type'] == place]
        if not sub.empty:
            fig.add_trace(go.Scatter(
                x=sub['timestamp'], y=sub['speed_kmh_smooth'],
                mode='lines', name=place,
                line=dict(color=color, width=2),
                fill='tozeroy',
                hovertemplate=(
                    f"<b>{place}</b><br>"
                    "Jour: %{x|%A %d-%m}<br>"
                    "Heure: %{x|%H:%M}<br>"
                    "Vitesse: %{y:.1f} km/h<extra></extra>"
                ),
                visible=True
            ))

    # --- Subplot 2 : Timeline des moves ---
    for _, mv in moves.iterrows():
        fig.add_trace(go.Scatter(
            x=[mv['start_time'], mv['end_time']],
            y=[1, 1],
            mode='lines',
            line=dict(color='orange', width=8),
            name=f"{mv['origin_type']} → {mv['destination_type']}",
            hovertemplate=(
                f"<b>Move</b><br>"
                f"{mv['origin_type']} → {mv['destination_type']}<br>"
                f"Durée: {mv['duration_s']/60:.1f} min<br>"
                f"Distance: {mv['dist_m']:.1f} m<extra></extra>"
            ),
            showlegend=True
        ), row=2, col=1)

    # --- Mise en forme générale ---
    fig.update_layout(
        title=f"Vitesse & Déplacements – ROLLING ({window_min} min)<br>"
              f"<span style='font-size:14px'>Total déplacements : {total_moves}, "
              f"Distance : {total_dist/1000:.1f} km, Durée : {total_duration:.1f} min</span>",
        xaxis=dict(title="Heure", tickformat="%d-%m %H:%M"),
        yaxis=dict(title="Vitesse (km/h)"),
        yaxis2=dict(title="Déplacements", showticklabels=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=50),
        height=700
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')


# def plot_rolling_speed_with_place(df_all, stops_df, window_min=10, by_day=False):
#     """
#     Trace la vitesse moyenne lissée avec :
#     - couleur différente pour chaque jour ou selon weekday/weekend
#     - style de ligne différent selon place_type (Home, Work, autre)
#     - option : un graphe global ou un graphe par jour

#     Returns:
#         str: HTML (Plotly)
#     """
#     # 1. Lissage de la vitesse
#     df = df_all.copy()
#     df['timestamp'] = pd.to_datetime(df['timestamp'])
#     df = df.set_index('timestamp').resample('1T').mean().interpolate()
#     df['speed_smooth'] = df['speed_kmh'].rolling(window=window_min, min_periods=1).mean()
#     df = df.reset_index()

#     # 2. Attribution des types de lieu (Home / Work / autre)
#     df['place_type'] = 'autre'
#     for _, stop in stops_df.iterrows():
#         import pytz

#         # Assure-toi que 'Europe/Paris' est utilisé pour tout
#         paris_tz = pytz.timezone("Europe/Paris")
#         start = stop['start_time']
#         end = stop['end_time']

#         if start.tzinfo is None:
#             start = paris_tz.localize(start)
#         if end.tzinfo is None:
#             end = paris_tz.localize(end)
#         if start.tzinfo is not None:
#             start = start.tz_localize(None)
#         if end.tzinfo is not None:
#             end = end.tz_localize(None)

#         mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)

#         df.loc[mask, 'place_type'] = stop['place_type']

#     # 3. Ajouter colonnes jour et weekend/weekday
#     df['jour'] = df['timestamp'].dt.date
#     df['weekday_name'] = df['timestamp'].dt.day_name()
#     df['day_type'] = df['timestamp'].dt.weekday.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')

#     if by_day:
#         # Tracé par jour séparé
#         html_parts = []
#         for day in sorted(df['jour'].unique()):
#             df_day = df[df['jour'] == day]
#             fig = px.line(
#                 df_day,
#                 x='timestamp',
#                 y='speed_smooth',
#                 color='place_type',
#                 line_dash='place_type',
#                 title=f"Vitesse lissée ({window_min} min) – {day} ({df_day['weekday_name'].iloc[0]})",
#                 labels={
#                     'speed_smooth': 'Vitesse (km/h)',
#                     'timestamp': 'Heure',
#                     'place_type': 'Type de lieu'
#                 }
#             )
#             fig.update_xaxes(dtick=600000, tickformat="%H:%M")
#             fig.update_layout(
#                 margin=dict(l=40, r=40, t=50, b=40),
#                 legend=dict(title="Type de lieu")
#             )
#             html_parts.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))
#         return "<br><br>".join(html_parts)

#     else:
#         # Graphique global avec couleur selon weekday/weekend et dash par type
#         fig = px.line(
#             df,
#             x='timestamp',
#             y='speed_smooth',
#             color='day_type',
#             line_dash='place_type',
#             title=f"Vitesse lissée ({window_min} min) – Weekday vs Weekend",
#             labels={
#                 'speed_smooth': 'Vitesse (km/h)',
#                 'timestamp': 'Heure',
#                 'day_type': 'Jour',
#                 'place_type': 'Type de lieu'
#             }
#         )
#         fig.update_xaxes(dtick=600000, tickformat="%H:%M")
#         fig.update_layout(
#             margin=dict(l=50, r=50, t=50, b=50),
#             legend=dict(title="Jour", orientation="h", y=1.02, x=1, xanchor="right"),
#         )
#         return fig.to_html(full_html=False, include_plotlyjs='cdn')

