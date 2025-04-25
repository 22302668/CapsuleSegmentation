from datetime import datetime
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import os
import folium
from branca.element import Template, MacroElement
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
import hdbscan
import umap
from sklearn.manifold import TSNE
import plotly.express as px
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sqlalchemy import text

#CONFIGURATION
report_path = os.path.join("data", "rapport_complet_HDBSCAN.html")
map_path = os.path.join("data", "carte_interactive_HDBSCAN.html")
umap_html_path = os.path.join("data", "umap_interactif.html")

(report_path, os.path.exists(os.path.dirname(report_path)))

#Chargement des variables d'environnement
load_dotenv()
url = f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DB')}"
engine = create_engine(url)

#Lecture depuis la table clean_gps
with engine.connect() as conn:
    df = pd.read_sql_query(text("SELECT * FROM clean_gps WHERE participant_virtual_id = '9999938'"), con=conn)

#Conversion du timestamp
df['timestamp'] = pd.to_datetime(df['time'], utc=True)  #convertit en UTC
df['timestamp'] = df['timestamp'].dt.tz_convert('Europe/Paris')  #convertit en heure locale France
df = df.drop(columns=['time'])
df = df.sort_values(by='timestamp').reset_index(drop=True)

#TEMPS & DISTANCE
df['time_diff_s'] = df['timestamp'].diff().dt.total_seconds()
distances = [None]
for i in range(1, len(df)):
    p1 = (df.loc[i-1, 'lat'], df.loc[i-1, 'lon'])
    p2 = (df.loc[i, 'lat'], df.loc[i, 'lon'])
    distances.append(geodesic(p1, p2).meters)
df['dist_m'] = distances
df['speed_kmh'] = df['dist_m'] / df['time_diff_s'] * 3.6
df['speed_kmh'] = df['speed_kmh'].replace([np.inf, -np.inf], np.nan)
df = df[df['speed_kmh'] <= 150]

#rolling moyenne
df['speed_kmh_smooth'] = df['speed_kmh'].rolling(window=5, min_periods=1, center=True).mean()

#CLUSTERING HDBSCAN
#Ajout de variables
df['hour'] = df['timestamp'].dt.hour
df['weekday_num'] = df['timestamp'].dt.weekday
df['accel'] = df['speed_kmh_smooth'].diff() / df['time_diff_s']  #estimation de l'accélération
df['accel'] = df['accel'].replace([np.inf, -np.inf], np.nan)

#Sélection des features pour clustering
df_ml = df[['speed_kmh_smooth', 'dist_m', 'time_diff_s', 'hour', 'weekday_num', 'accel']].dropna()

#STANDARDISATION
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_ml)

#CLUSTERING HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=5)
df.loc[df_ml.index, 'cluster_behavior'] = clusterer.fit_predict(X_scaled)

#PROJECTION UMAP 2D
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding = umap_model.fit_transform(X_scaled)
df_umap = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"], index=df_ml.index)
df[['UMAP1', 'UMAP2']] = df_umap

#UMAP INTERACTIF AVEC PLOTLY
df_plotly = df.loc[df['UMAP1'].notna()].copy()
df_plotly['cluster_behavior'] = df_plotly['cluster_behavior'].astype(int)

fig_umap_hover = px.scatter(
    df_plotly, 
    x='UMAP1', 
    y='UMAP2', 
    color='cluster_behavior',
    hover_data={
        'timestamp': True,
        'speed_kmh': ':.2f',
        'dist_m': ':.2f',
        'time_diff_s': ':.2f',
        'cluster_behavior': True,
        'UMAP1': False,
        'UMAP2': False
    },
    title="Projection UMAP interactive des comportements",
    template="plotly_white"
)
fig_umap_hover.update_traces(marker=dict(size=5, opacity=0.7))


#Cluster name mapping
cluster_labels = df.groupby('cluster_behavior')['speed_kmh_smooth'].mean()
name_map = {}
for cluster, speed in cluster_labels.items():
    if cluster == -1:
        name_map[cluster] = "Bruit"
    elif speed < 1:
        name_map[cluster] = "Arrêt prolongé"
    elif speed < 5:
        name_map[cluster] = "Marche"
    elif speed < 15:
        name_map[cluster] = "Vélo"
    else:
        name_map[cluster] = "Transport rapide"
df['cluster_label'] = df['cluster_behavior'].map(name_map)

perplexity_val = min(30, len(X_scaled) - 1)  # t-SNE nécessite perplexity < n_samples
embedding_tsne = TSNE(n_components=2, perplexity=perplexity_val, learning_rate=100, random_state=42).fit_transform(X_scaled)

#VISUALISATION t-SNE DES CLUSTERS
df_tsne = pd.DataFrame(embedding_tsne, columns=["TSNE1", "TSNE2"], index=df_ml.index)
df[['TSNE1', 'TSNE2']] = df_tsne

#ASSIGNATION COULEURS CLUSTERS
unique_clusters = df['cluster_behavior'].dropna().unique()
palette = sns.color_palette("tab20", len(unique_clusters)).as_hex()
cluster_color_map = {c: palette[i % len(palette)] for i, c in enumerate(unique_clusters)}
cluster_color_map[-1] = "gray"
df['cluster_color'] = df['cluster_behavior'].map(lambda x: cluster_color_map.get(x, "gray"))

#CLASSIFICATION PAR SEUILS
bins = [-1, 0.5, 5, 15, 150]
labels = ['stop', 'slow_walk', 'fast_walk', 'transport']
df['movement_type'] = pd.cut(df['speed_kmh'], bins=bins, labels=labels)
df['movement_type'] = df['movement_type'].cat.add_categories(['unknown']).fillna('unknown')

#INFOS TEMPORELLES
df['hour'] = df['timestamp'].dt.hour
df['weekday'] = df['timestamp'].dt.day_name()
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

#GEOMETRIE POUR FOLIUM
df['geometry'] = df.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')

def detect_stops_in_noise(df, speed_threshold=1.0, min_duration_s=60):
    #On garde uniquement les points considérés comme bruit avec une vitesse faible
    df_noise = df[(df['cluster_behavior'] == -1) & (df['speed_kmh'] < speed_threshold)].copy()
    
    #On identifie les groupes d'arrêts consécutifs en regardant l'écart temporel
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

    # On filtre les vrais arrêts longs
    stops_summary = stops_summary[stops_summary['duration_s'] >= min_duration_s]
    return stops_summary

#FIGURES POUR RAPPORT
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

fig1, ax1 = plt.subplots(figsize=(8,4))
df['speed_kmh'].clip(upper=80).hist(bins=60, ax=ax1)
ax1.set_title("Distribution des vitesses")
img_speed = fig_to_base64(fig1)

fig_low, ax_low = plt.subplots(figsize=(8, 4))
df[df['speed_kmh'] <= 10]['speed_kmh'].hist(bins=30, ax=ax_low)
ax_low.set_title("Distribution des vitesses ≤ 10 km/h")
img_speed_low = fig_to_base64(fig_low)

fig_high, ax_high = plt.subplots(figsize=(8, 4))
df[df['speed_kmh'] > 10]['speed_kmh'].hist(bins=30, ax=ax_high)
ax_high.set_title("Distribution des vitesses > 10 km/h")
img_speed_high = fig_to_base64(fig_high)

fig2, ax2 = plt.subplots(figsize=(10, 4))
df.groupby('hour')['speed_kmh'].mean().plot(kind='bar', ax=ax2)
ax2.set_title("Vitesse moyenne par heure")
img_hour = fig_to_base64(fig2)

fig3, ax3 = plt.subplots(figsize=(10, 4))
df.groupby('weekday')['speed_kmh'].mean().reindex(weekday_order).plot(kind='bar', ax=ax3)
ax3.set_title("Vitesse moyenne par jour")
img_day = fig_to_base64(fig3)

fig4, ax4 = plt.subplots(figsize=(6, 4))
df['movement_type'].value_counts().plot(kind='bar', ax=ax4, color=['red', 'orange', 'blue', 'gray'])
ax4.set_title("Répartition des types de mouvement")
img_type = fig_to_base64(fig4)

fig5, ax5 = plt.subplots(figsize=(6, 4))
df['cluster_behavior'].value_counts().sort_index().plot(kind='bar', ax=ax5)
ax5.set_title("Répartition des comportements détectés (HDBSCAN)")
img_cluster = fig_to_base64(fig5)

fig6, ax6 = plt.subplots(figsize=(8, 6))
for label in sorted(df['cluster_behavior'].dropna().unique()):
    subset = df[df['cluster_behavior'] == label]
    ax6.scatter(subset['UMAP1'], subset['UMAP2'], s=10, label=f'Cluster {int(label)}', alpha=0.6)
ax6.set_title("Projection UMAP des comportements (HDBSCAN)")
ax6.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
img_umap = fig_to_base64(fig6)

fig7, ax7 = plt.subplots(figsize=(8, 6))
for label in sorted(df['cluster_behavior'].dropna().unique()):
    subset = df[df['cluster_behavior'] == label]
    ax7.scatter(subset['TSNE1'], subset['TSNE2'], s=10, label=f'Cluster {int(label)}', alpha=0.6)
ax7.set_title("Projection t-SNE des comportements (HDBSCAN)")
ax7.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
img_tsne = fig_to_base64(fig7)

# Analyse des "stops" cachés dans le bruit
stops_in_noise = detect_stops_in_noise(df)
stops_html = stops_in_noise[['start_time', 'end_time', 'duration_s', 'lat', 'lon', 'speed_kmh_mean', 'n_points']].to_html(index=False)
stop_coords = stops_in_noise[['lat', 'lon', 'start_time', 'end_time', 'duration_s']].values.tolist()

# Fixer les couleurs manuellement (au lieu de sns)
cluster_color_map = {
    -1: "gray",            # Bruit
    0: "green",            # Transport rapide
    1: "blue",             # Vélo
    2: "orange",           # Marche
    3: "red"               # Arrêt prolongé
}
df['cluster_color'] = df['cluster_behavior'].map(lambda x: cluster_color_map.get(x, "gray"))

#CARTE INTERACTIVE
map_center = [df['lat'].mean(), df['lon'].mean()]
m = folium.Map(location=map_center, zoom_start=13)

#Points GPS classiques
for _, row in df.iterrows():
    tooltip = folium.Tooltip(f"Comportement: {row['cluster_label']}")
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=3,
        color=row.get('cluster_color', 'gray'),
        fill=True,
        fill_color=row.get('cluster_color', 'gray'),
        fill_opacity=0.6,
        tooltip=tooltip,
        popup=f"{row['timestamp']}<br>Cluster: {row.get('cluster_behavior')}<br>{row['speed_kmh']:.2f} km/h" if pd.notna(row['speed_kmh']) else ""
    ).add_to(m)

#Ajout des stops dans le bruit (en rouge vif )
for lat, lon, start, end, duration in stop_coords:
    folium.Marker(
        location=[lat, lon],
        icon=folium.Icon(color="red", icon="pause", prefix="fa"),
        popup=f"<b>Stop détecté</b><br><b>Début:</b> {start}<br><b>Fin:</b> {end}<br><b>Durée:</b> {int(duration)} s"
    ).add_to(m)

legend_html = """
{% macro html(this, kwargs) %}
<div style='position: fixed; bottom: 30px; left: 30px; width: 220px; height: 200px;
     background-color: white; z-index:9999; font-size:14px;
     border:2px solid gray; padding: 10px;'>
<b>🗌️ Légende :</b><br>
<span style='color:red;'>⬤</span> Arrêt prolongé<br>
<span style='color:orange;'>⬤</span> Marche<br>
<span style='color:blue;'>⬤</span> Vélo<br>
<span style='color:green;'>⬤</span> Transport rapide<br>
<span style='color:gray;'>⬤</span> Bruit / Inconnu<br>
<span style='color:red;'>📍</span> Stop dans le bruit (détecté)<br>
</div>
{% endmacro %}
"""

legend = MacroElement()
legend._template = Template(legend_html)
m.get_root().add_child(legend)
m.save(map_path)


# === RAPPORT HTML ===
generation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
html = f"""
<h1>📍 Rapport GPS avec Détection HDBSCAN</h1>
<p><strong>Date :</strong> {generation_date}</p>

<h2>🚗 Statistiques de vitesse (km/h)</h2>
{df['speed_kmh'].dropna().describe().to_frame().to_html()}

<h2>🏃‍♂️ Répartition des types de mouvement</h2>
{df['movement_type'].value_counts().to_frame(name='nombre_points').to_html()}

<h2>🧠 Répartition des clusters détectés automatiquement (HDBSCAN)</h2>
{df['cluster_behavior'].value_counts().to_frame(name='nombre_points').to_html()}

<h2>📊 Visualisations</h2>
<h4>Distribution des vitesses</h4><img src="data:image/png;base64,{img_speed}" width="600"/><br>
<h4>Vitesses ≤ 10 km/h</h4><img src="data:image/png;base64,{img_speed_low}" width="600"/><br>
<h4>Vitesses > 10 km/h</h4><img src="data:image/png;base64,{img_speed_high}" width="600"/><br>
<h4>Vitesse par heure</h4><img src="data:image/png;base64,{img_hour}" width="600"/><br>
<h4>Vitesse par jour</h4><img src="data:image/png;base64,{img_day}" width="600"/><br>
<h4>Types de déplacement</h4><img src="data:image/png;base64,{img_type}" width="600"/><br>
<h4>Clusters (comportements détectés)</h4><img src="data:image/png;base64,{img_cluster}" width="600"/><br>
<h4>Projection UMAP des clusters</h4><img src="data:image/png;base64,{img_umap}" width="700"/><br>
<h4>Projection t-SNE des clusters</h4><img src="data:image/png;base64,{img_tsne}" width="700"/><br> 
<iframe src="umap_interactif.html" width="100%" height="600" style="border:none;"></iframe> 

<h2>🛑 Stops détectés dans le bruit (Cluster -1)</h2>
<p>Ces arrêts prolongés n’ont pas été identifiés comme un cluster par HDBSCAN, mais correspondent à une vitesse très faible (< 1 km/h) sur une durée significative (≥ 60 s).</p>
{stops_html}

<iframe src="carte_interactive_HDBSCAN.html" width="100%" height="600" style="border:none;"></iframe> 

"""

umap_html_path = os.path.join("data", "umap_interactif.html")
fig_umap_hover.write_html(umap_html_path)

with open(report_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\n✔️ Rapport généré : {report_path}")
