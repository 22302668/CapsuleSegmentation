from datetime import datetime
import os
import folium
from folium.plugins import MarkerCluster, HeatMap, MiniMap
import base64
from io import BytesIO
import seaborn as sns
import matplotlib.pyplot as plt

def generate_interactive_map(df):
    """
    Génère une carte Folium enrichie :
    - Filtres par date
    - Filtres par type de mouvement
    - Heatmap de densité
    - Trajet complet affichable
    - Zoom automatique sur les groupes visibles
    """
    if df.empty:
        return ""

    center_lat = df['lat'].mean()
    center_lon = df['lon'].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='cartodbpositron')

    # MiniMap
    minimap = MiniMap(toggle_display=True)
    m.add_child(minimap)

    # Couleurs
    colors = {
        'stop': 'red',
        'slow_walk': 'orange',
        'fast_walk': 'blue',
        'transport': 'green',
        'unknown': 'gray'
    }

    # Ajouter colonnes auxiliaires
    df['date_only'] = df['timestamp'].dt.date

    feature_groups = []

    # --------
    # Groupe 1: Par mouvement_type
    # --------
    for movement_type, group in df.groupby('movement_type'):
        fg_movement = folium.FeatureGroup(name=f"Type: {movement_type} ({len(group)})", show=False)
        bounds = []

        for _, row in group.iterrows():
            color = colors.get(row['movement_type'], 'black')
            folium.CircleMarker(
                location=(row['lat'], row['lon']),
                radius=4,
                color=color,
                fill=True,
                fill_opacity=0.8,
                popup=folium.Popup(
                    f"""
                    <b>Type:</b> {row['movement_type']}<br>
                    <b>Vitesse:</b> {row['speed_kmh']:.2f} km/h<br>
                    <b>Date:</b> {row['timestamp'].date()}<br>
                    <b>Heure:</b> {row['timestamp'].time()}
                    """,
                    max_width=300
                )
            ).add_to(fg_movement)
            bounds.append((row['lat'], row['lon']))

        if bounds:
            m.fit_bounds(bounds)

        fg_movement.add_to(m)
        feature_groups.append(fg_movement)

    # --------
    # Groupe 2: Par jour
    # --------
    for date, group in df.groupby('date_only'):
        fg_date = folium.FeatureGroup(name=f"Jour: {date} ({len(group)})", show=False)
        bounds = []

        for _, row in group.iterrows():
            color = colors.get(row['movement_type'], 'black')
            folium.CircleMarker(
                location=(row['lat'], row['lon']),
                radius=4,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=folium.Popup(
                    f"""
                    <b>Date:</b> {row['timestamp'].date()}<br>
                    <b>Heure:</b> {row['timestamp'].time()}<br>
                    <b>Type:</b> {row['movement_type']}<br>
                    <b>Vitesse:</b> {row['speed_kmh']:.2f} km/h
                    """,
                    max_width=300
                )
            ).add_to(fg_date)
            bounds.append((row['lat'], row['lon']))

        if bounds:
            m.fit_bounds(bounds)

        fg_date.add_to(m)
        feature_groups.append(fg_date)

    # --------
    # Trajet complet
    # --------
    if not df[['lat', 'lon']].dropna().empty:
        full_path = df[['lat', 'lon']].dropna().values.tolist()
        fg_full_trajet = folium.FeatureGroup(name="🛣️ Trajet complet", show=True)
        folium.PolyLine(
            full_path,
            color="black",
            weight=3,
            opacity=0.7,
            tooltip="Trajet complet"
        ).add_to(fg_full_trajet)
        fg_full_trajet.add_to(m)

    # --------
    # Heatmap
    # --------
    if not df[['lat', 'lon']].dropna().empty:
        heat_data = df[['lat', 'lon']].dropna().values.tolist()
        fg_heatmap = folium.FeatureGroup(name="🔥 Heatmap Densité", show=False)
        HeatMap(heat_data, radius=10, blur=15, max_zoom=13).add_to(fg_heatmap)
        fg_heatmap.add_to(m)

    # --------
    # Layer Control
    # --------
    folium.LayerControl(collapsed=False).add_to(m)

    return m.get_root().render()

def generate_activity_heatmap(crosstab):
    """
    Génère une heatmap en base64 du tableau croisé cluster-activité.
    """
    plt.figure(figsize=(14, 6))
    sns.heatmap(crosstab, annot=True, fmt="d", cmap="YlGnBu", linewidths=0.5)
    plt.title("Heatmap des correspondances cluster/activité")
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return encoded

def generate_full_report(df, stops_summary, figures_base64, activities_crosstab=None):
    os.makedirs("data", exist_ok=True)
    report_path = os.path.join("data", "rapport_complet_HDBSCAN.html")
    generation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Générer carte + heatmap si activités présentes
    map_html = generate_interactive_map(df)
    heatmap_base64 = generate_activity_heatmap(activities_crosstab) if activities_crosstab is not None else None

    html = f""" 
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Rapport GPS - HDBSCAN</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h4 {{ color: #2c3e50; }}
            img {{ margin-bottom: 20px; }}
            .table-container {{ overflow-x: auto; }}
        </style>
    </head>

    <body>

    <h1>📍 Rapport d'Analyse GPS avec HDBSCAN</h1>
    <p><strong>Date de génération :</strong> {generation_date}</p>

    <h2>🧭 Table des matières</h2>
    <ul>
      <li><a href="#map">🗺️ Carte interactive</a></li>
      <li><a href="#stats_vitesse">🚗 Statistiques de vitesse</a></li>
      <li><a href="#types_mouvements">🏃‍♂️ Types de mouvements</a></li>
      <li><a href="#clusters_detectes">🧠 Clusters détectés</a></li>
      <li><a href="#evaluation_clustering">📊 Évaluation du clustering</a></li>
      <li><a href="#stops_bruit">🛑 Stops détectés</a></li>
    </ul>

    <hr>

    <h2 id="map">🗺️ Carte interactive des trajectoires</h2>
    {map_html}

    <hr>

    <h2 id="stats_vitesse">🚗 Statistiques de vitesse (km/h)</h2>
    <div class="table-container">{df['speed_kmh'].dropna().describe().to_frame().to_html()}</div>

    <hr>

    <h2 id="types_mouvements">🏃‍♂️ Répartition des types de mouvement</h2>
    <div class="table-container">{df['movement_type'].value_counts().to_frame(name='nombre_points').to_html()}</div>

    <hr>

    <h2 id="clusters_detectes">🧠 Répartition des clusters détectés automatiquement (HDBSCAN)</h2>
    <div class="table-container">{df['cluster_behavior'].value_counts().to_frame(name='nombre_points').to_html()}</div>

    <h2>🧠 Répartition après étiquetage KMeans</h2>
    <div class="table-container">{df['cluster_label'].value_counts().to_frame(name='nombre_points').to_html()}</div>

    <hr>

    <h2 id="evaluation_clustering">📊 Évaluation du clustering HDBSCAN</h2>
    <h4>Silhouette / Davies-Bouldin / Calinski-Harabasz</h4>
    <img src="data:image/png;base64,{figures_base64['scores_hdbscan']}" width="700"/>

    <hr>

    <h2>📈 Analyses complémentaires</h2>
    <h4>Distribution des vitesses</h4><img src="data:image/png;base64,{figures_base64['distribution_vitesse']}" width="600"/><br>
    <h4>Vitesses ≤ 10 km/h</h4><img src="data:image/png;base64,{figures_base64['distribution_vitesse_basse']}" width="600"/><br>
    <h4>Vitesses > 10 km/h</h4><img src="data:image/png;base64,{figures_base64['distribution_vitesse_haute']}" width="600"/><br>
    <h4>Vitesse moyenne par heure</h4><img src="data:image/png;base64,{figures_base64['vitesse_par_heure']}" width="600"/><br>
    <h4>Vitesse moyenne par jour</h4><img src="data:image/png;base64,{figures_base64['vitesse_par_jour']}" width="600"/><br>
    <h4>Types de déplacement détectés</h4><img src="data:image/png;base64,{figures_base64['types_de_mouvement']}" width="600"/><br>
    <h4>Clusters détectés</h4><img src="data:image/png;base64,{figures_base64['clusters_detectes']}" width="600"/><br>
    """

    if heatmap_base64:
        html += f"""
        <h4>🔥 Heatmap correspondance Clusters / Activités</h4>
        <img src="data:image/png;base64,{heatmap_base64}" width="900"/><br>
        """

    html += f"""
    <hr>

    <h2 id="stops_bruit">🛑 Stops détectés dans le bruit (Cluster -1)</h2>
    <div class="table-container">{stops_summary.to_html(index=False)}</div>
    """

    if activities_crosstab is not None:
        html += f"""
        <hr>
        <h2>🧩 Vérification des Activités détectées</h2>
        <p>Tableau croisé entre clusters détectés et activités enregistrées :</p>
        <div class="table-container">{activities_crosstab.to_html(border=0)}</div>
        """

    html += """
    </body>
    </html>
    """

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✔️ Rapport proprement généré ici : {report_path}")
