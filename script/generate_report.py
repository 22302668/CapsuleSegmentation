from datetime import datetime
import os
import folium
from folium.plugins import HeatMap, MiniMap
import pandas as pd
import matplotlib.pyplot as plt

def generate_interactive_map(df, stops_summary, grouped_stops, classified_stops):
    center_lat = df['lat'].mean()
    center_lon = df['lon'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='cartodbpositron')
    m.add_child(MiniMap(toggle_display=True))

    if not df[['lat', 'lon']].dropna().empty:
        path = df[['lat', 'lon']].dropna().values.tolist()
        folium.PolyLine(path, color='black', weight=3, opacity=0.6).add_to(
            folium.FeatureGroup(name="🚣️ Trajet complet", show=True).add_to(m)
        )

    heat_data = df[['lat', 'lon']].dropna().values.tolist()
    if heat_data:
        HeatMap(heat_data, radius=10, blur=15).add_to(
            folium.FeatureGroup(name="🔥 Heatmap Densité", show=False).add_to(m)
        )

    if stops_summary is not None and not stops_summary.empty:
        stops_summary['day'] = pd.to_datetime(stops_summary['start_time']).dt.date
        fg_all = folium.FeatureGroup(name=f"🚩 Tous les stops ({len(stops_summary)})", show=True)

        for _, row in stops_summary.iterrows():
            duration_min = row['duration_s'] / 60
            radius = min(10, max(3, duration_min / 2))
            color = (
                'green' if duration_min < 5 else
                'orange' if duration_min < 15 else
                'red'
            )
            popup = folium.Popup(
                f"<b>Début:</b> {row['start_time']}<br>"
                f"<b>Fin:</b> {row['end_time']}<br>"
                f"<b>Durée:</b> {int(row['duration_s'] // 60)} min {int(row['duration_s'] % 60)} sec",
                max_width=300
            )
            folium.CircleMarker(
                location=(row['lat'], row['lon']),
                radius=radius,
                color=color,
                fill=True,
                fill_opacity=0.85,
                popup=popup
            ).add_to(fg_all)
        fg_all.add_to(m)

        for day, group in stops_summary.groupby('day'):
            fg_day = folium.FeatureGroup(name=f"📆 Stops du {day}({len(group)})", show=False)
            for _, row in group.iterrows():
                duration_min = row['duration_s'] / 60
                radius = min(10, max(3, duration_min / 2))
                color = (
                    'green' if duration_min < 5 else
                    'orange' if duration_min < 15 else
                    'red'
                )
                popup = folium.Popup(
                    f"<b>Date:</b> {row['start_time'].date()}<br>"
                    f"<b>Début:</b> {row['start_time'].time()}<br>"
                    f"<b>Fin:</b> {row['end_time'].time()}<br>"
                    f"<b>Durée:</b> {int(row['duration_s'] // 60)} min {int(row['duration_s'] % 60)} sec",
                    max_width=300
                )
                folium.CircleMarker(
                    location=(row['lat'], row['lon']),
                    radius=radius,
                    color=color,
                    fill=True,
                    fill_opacity=0.85,
                    popup=popup
                ).add_to(fg_day)
            fg_day.add_to(m)

    if grouped_stops is not None and not grouped_stops.empty:
        fg_grouped = folium.FeatureGroup(name=f"📍 Stops regroupés ({len(grouped_stops)})", show=True)
        for _, row in grouped_stops.iterrows():
            popup = folium.Popup(
                f"<b>STOP REGROUPÉ</b><br>"
                f"Début : {row['start_time']}<br>"
                f"Fin : {row['end_time']}<br>"
                f"Durée : {int(row['duration_s'])} s<br>"
                f"Points fusionnés : {row.get('group_size', 'N/A')}",
                max_width=300
            )
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=6,
                color='black',
                fill=True,
                fill_color='white',
                fill_opacity=1,
                popup=popup
            ).add_to(fg_grouped)
        fg_grouped.add_to(m)

    if classified_stops is not None and 'place_type' in classified_stops.columns:
        fg_place_type = folium.FeatureGroup(name="🏠 Lieux classifiés (Home/Work)", show=True)
        for _, row in classified_stops.iterrows():
            color = {
                'Home': 'blue',
                'Work': 'purple',
                'Unknown': 'gray'
            }.get(row['place_type'], 'black')
            intervals = "".join(
                f"<li>{s} → {e}</li>"
                for s, e in zip(row['merged_intervals'], row['merged_ends'])
            )
            popup = folium.Popup(
                f"<b>Lieu détecté :</b> <span style='color:{color}'>{row['place_type']}</span><br>"
                f"<b>Durée totale :</b> {int(row['duration_s'] // 60)} min<br>"
                f"<b>⏱️ Intervalles :</b><ul>{intervals}</ul>",
                max_width=300
            )

            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=9,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                popup=popup
            ).add_to(fg_place_type)
        fg_place_type.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m.get_root().render()

def generate_full_report(df, stops_summary, figures_base64,grouped_stops, classified_stops):
    os.makedirs("data", exist_ok=True)
    report_path = os.path.join("data", "rapport_stops_movingpandas.html")
    generation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    map_html = generate_interactive_map(df, stops_summary, grouped_stops, classified_stops)

    nb_points = len(df)
    nb_stops = len(stops_summary)
    total_duration_s = stops_summary['duration_s'].sum() if 'duration_s' in stops_summary.columns else 0
    total_duration_min = round(total_duration_s / 60, 1)
    total_distance_km = df['dist_m'].sum() / 1000 if 'dist_m' in df.columns else None
    start_time = df['timestamp'].min()
    end_time = df['timestamp'].max()
    duration_total = end_time - start_time
    nb_days = df['timestamp'].dt.date.nunique()
    avg_speed = round(df['speed_kmh'].mean(), 2)
    max_speed = round(df['speed_kmh'].max(), 2)
    avg_points_per_day = int(df.groupby(df['timestamp'].dt.date).size().mean())
    sampling_interval = round(df['timestamp'].diff().dt.total_seconds().mean(), 1)

    html = f"""
    <html>
    <head>
        <meta charset=\"UTF-8\">
        <title>Rapport GPS - MovingPandas</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h4 {{ color: #2c3e50; }}
            img {{ margin-bottom: 20px; }}
            .table-container {{ overflow-x: auto; }}
        </style>
    </head>
    <body>

    <h1>📍 Rapport d'Analyse GPS avec MovingPandas</h1>
    <p><strong>Date de génération :</strong> {generation_date}</p>

    <h2>📊 Résumé global</h2>
    <ul>
        <li><strong>Nombre total de points :</strong> {nb_points}</li>
        <li><strong>Nombre de stops détectés :</strong> {nb_stops}</li>
        <li><strong>Durée totale des stops :</strong> {total_duration_min} minutes</li>
        {f'<li><strong>Distance totale parcourue :</strong> {total_distance_km:.2f} km</li>' if total_distance_km else ''}
        <li><strong>Durée totale d'enregistrement :</strong> {duration_total} (du {start_time.date()} au {end_time.date()})</li>
        <li><strong>Nombre de jours couverts :</strong> {nb_days} jour(s)</li>
        <li><strong>Vitesse moyenne :</strong> {avg_speed} km/h</li>
        <li><strong>Vitesse maximale :</strong> {max_speed} km/h</li>
        <li><strong>Nombre moyen de points par jour :</strong> {avg_points_per_day}</li>
        <li><strong>Fréquence moyenne d’échantillonnage :</strong> un point toutes les {sampling_interval} secondes</li>
    </ul>

    <h2>🧭 Table des matières</h2>
    <ul>
      <li><a href=\"#map\">🗺️ Carte interactive</a></li>
      <li><a href=\"#stats_vitesse\">🚗 Statistiques de vitesse</a></li>
      <li><a href=\"#stops\">🛑 Stops détectés</a></li>
      <li><a href=\"#frequence\">📅 Fréquence des points GPS</a></li>
    </ul>

    <hr>
    <section style=\"position: relative; margin-top: 40px; z-index: 0;\">
    <h2 id=\"map\">🗺️ Carte interactive des trajectoires</h2>
    {map_html}
    </section>

    <h2 id=\"stats_vitesse\">🚗 Statistiques de vitesse (km/h)</h2>
    <div class=\"table-container\">{df['speed_kmh'].dropna().describe().to_frame().to_html()}</div>

    <hr>
    <h2 id=\"stops\">🛑 Stops détectés avec MovingPandas</h2>
    <div class=\"table-container\">{stops_summary.to_html(index=False)}</div>

    <hr>
    <h2>📈 Analyses complémentaires</h2>
    <h4>Distribution des vitesses</h4><img src=\"data:image/png;base64,{figures_base64['distribution_vitesse']}\" width=\"600\"/><br>
    <h4>Vitesses ≤ 10 km/h</h4><img src=\"data:image/png;base64,{figures_base64['distribution_vitesse_basse']}\" width=\"600\"/><br>
    <h4>Vitesses > 10 km/h</h4><img src=\"data:image/png;base64,{figures_base64['distribution_vitesse_haute']}\" width=\"600\"/><br>
    <h4>Vitesse moyenne par heure</h4><img src=\"data:image/png;base64,{figures_base64['vitesse_par_heure']}\" width=\"600\"/><br>
    <h4>Vitesse moyenne par jour</h4><img src=\"data:image/png;base64,{figures_base64['vitesse_par_jour']}\" width=\"600\"/><br>

    """

    if 'stop_timeline' in figures_base64:
        html += f"""
        <hr>
        <h2>📊 Analyse temporelle des arrêts selon l'activité détectée</h2>
        <h4>Durée des arrêts par heure, colorée par activité</h4>
        <img src="data:image/png;base64,{figures_base64['stop_timeline']}" width="700"/><br>
        """

    if 'stop_mean_duration' in figures_base64:
        html += f"""
        <h4>Durée moyenne des arrêts par activité</h4>
        <img src="data:image/png;base64,{figures_base64['stop_mean_duration']}" width="700"/><br>
        """
    if grouped_stops is not None:
        html += "<h3>🧩 Stops regroupés (Bounding Box)</h3>"
        html += grouped_stops[['start_time', 'end_time', 'duration_s', 'lat', 'lon',  'group_size']].to_html(index=False)
    
    if classified_stops is not None and 'place_type' in classified_stops.columns:
        html += "<h3>🏷️ Lieux classifiés : Home, Work, Unknown</h3>"
        html += classified_stops[['start_time', 'end_time', 'duration_s', 'lat', 'lon', 'place_type']].to_html(index=False)

    if 'stop_heatmap' in figures_base64:
        html += f"""
        <h4>📅 Heatmap spatio-temporelle : durée cumulée des arrêts</h4>
        <img src="data:image/png;base64,{figures_base64['stop_heatmap']}" width="900"/><br>
        """
    
    if 'stop_weekday_vs_weekend' in figures_base64:
        html += f"""
        <hr>
        <h4>⏰ Comparaison des arrêts – Semaine vs Weekend</h4>
        <img src="data:image/png;base64,{figures_base64['stop_weekday_vs_weekend']}" width="800"/><br>
        """
        if 'stops_dispersion' in figures_base64:
            html += f"""
            <h4>Position des stops (durée en couleur)</h4>
            <img src="data:image/png;base64,{figures_base64['stops_dispersion']}" width="600"/><br>
            <h4>Durée des stops (en secondes)</h4>
            <img src="data:image/png;base64,{figures_base64['stops_duree']}" width="600"/><br>
            <h4>Nombre de stops par heure</h4>
            <img src="data:image/png;base64,{figures_base64['stops_par_heure']}" width="600"/><br>
            """
    if 'points_weekdays_vs_weekends' in figures_base64:
        html += f"""
        <hr>
        <h4>📊 Répartition des points GPS – Semaine vs Weekend</h4>
        <img src="data:image/png;base64,{figures_base64['points_weekdays_vs_weekends']}" width="800"/><br>
        """

    if 'points_par_jour' in figures_base64:
        html += f"""
        <hr>
        <h2 id="frequence">📅 Fréquence des points GPS</h2>
        <h4>Points par jour</h4><img src="data:image/png;base64,{figures_base64['points_par_jour']}" width="600"/><br>
        <h4>Points par heure</h4><img src="data:image/png;base64,{figures_base64['points_par_heure']}" width="600"/><br>
        <h4>Heatmap jour x heure</h4><img src="data:image/png;base64,{figures_base64['heatmap_frequence']}" width="900"/><br>
        """

    html += """
    </body>
    </html>
    """

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nRapport proprement généré ici : {report_path}")
