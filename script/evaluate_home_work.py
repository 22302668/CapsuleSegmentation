from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from io import BytesIO
import base64

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
