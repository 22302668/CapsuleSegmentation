from datetime import datetime
import folium
from folium.plugins import HeatMap, MiniMap
import pandas as pd
import geopandas as gpd

from detect_stops_and_analyze import generate_figures
from movingpandas_stop_detection import detect_stops_and_moves
from scikit_mobility import detect_stops_with_skmob
from evaluate_home_work import plot_rolling_speed
from dbscan_clustering              import cluster_stops_dbscan
from split_moves_stops import build_moves_summary

def generate_interactive_map(df, stops_summary, grouped_stops, final_stops):
    """
    Retourne le HTML d’une carte Folium pour un DataFrame donné.
    final_stops correspond aux arrêts déjà fusionnés / classifiés (Home/Work/autre).
    """
    center_lat = df['lat'].mean()
    center_lon = df['lon'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='cartodbpositron')
    m.add_child(MiniMap(toggle_display=True))

    # 1) Trajet complet
    if not df[['lat', 'lon']].dropna().empty:
        path = df[['lat', 'lon']].dropna().values.tolist()
        fg_trajet_global = folium.FeatureGroup(name="Trajet complet", show=True)
        folium.PolyLine(path, color='black', weight=3, opacity=0.6).add_to(fg_trajet_global)
        fg_trajet_global.add_to(m)

    # 2) Calque par jour
    if 'timestamp' in df.columns:
        df['day'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert("Europe/Paris").dt.date
        for day, group in df.groupby('day'):
            fg_jour = folium.FeatureGroup(name=f"Trajet {day}", show=False)
            coords = group[['lat', 'lon']].dropna().values.tolist()
            folium.PolyLine(coords, color='blue', weight=2, opacity=0.6).add_to(fg_jour)
            for _, row in group.iterrows():
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=2,
                    color='blue',
                    fill=True,
                    fill_opacity=0.4
                ).add_to(fg_jour)
            fg_jour.add_to(m)

    # 3) Heatmap densité
    heat_data = df[['lat', 'lon']].dropna().values.tolist()
    if heat_data:
        fg_heat = folium.FeatureGroup(name="Heatmap Densité", show=False)
        HeatMap(heat_data, radius=10, blur=15).add_to(fg_heat)
        fg_heat.add_to(m)

    # 4) Stops bruts
    if stops_summary is not None and not stops_summary.empty:
        stops_summary['day'] = pd.to_datetime(stops_summary['start_time']).dt.date
        fg_all = folium.FeatureGroup(name=f"Tous les stops ({len(stops_summary)})", show=True)
        for _, row in stops_summary.iterrows():
            duration_min = row['duration_s'] / 60
            radius = min(10, max(3, duration_min / 2))
            color = (
                'green' if duration_min < 5 else
                'orange' if duration_min < 15 else
                'red'
            )
            popup = folium.Popup(
                f"<b>Début :</b> {row['start_time']}<br>"
                f"<b>Fin :</b> {row['end_time']}<br>"
                f"<b>Durée :</b> {int(row['duration_s'] // 60)} min {int(row['duration_s'] % 60)} sec",
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

        # Stops par jour
        for day, group in stops_summary.groupby('day'):
            fg_day = folium.FeatureGroup(name=f"Stops du {day} ({len(group)})", show=False)
            for _, row in group.iterrows():
                duration_min = row['duration_s'] / 60
                radius = min(10, max(3, duration_min / 2))
                color = (
                    'green' if duration_min < 5 else
                    'orange' if duration_min < 15 else
                    'red'
                )
                popup = folium.Popup(
                    f"<b>Date :</b> {row['start_time'].date()}<br>"
                    f"<b>Début :</b> {row['start_time'].time()}<br>"
                    f"<b>Fin :</b> {row['end_time'].time()}<br>"
                    f"<b>Durée :</b> {int(row['duration_s'] // 60)} min {int(row['duration_s'] % 60)} sec",
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

    # 5) Stops regroupés (bounding box)
    if grouped_stops is not None and not grouped_stops.empty:
        fg_grouped = folium.FeatureGroup(name=f"Stops regroupés ({len(grouped_stops)})", show=True)
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

    # 6) Lieux classifiés (Home / Work / autre) → ici on utilise le DataFrame `final_stops`
    if final_stops is not None and 'place_type' in final_stops.columns:
        fg_place_type = folium.FeatureGroup(name="Lieux classifiés (Home/Work)", show=True)
        for _, row in final_stops.iterrows():
            color = {
                'Home': 'blue',
                'Work': 'green',   # par exemple, Work en vert
                'autre': 'gray'
            }.get(row['place_type'], 'black')
            popup = folium.Popup(
                f"<b>Lieu :</b> {row['place_type']}<br>"
                f"<b>Début :</b> {row['start_time']}<br>"
                f"<b>Fin :</b> {row['end_time']}<br>"
                f"<b>Durée :</b> {int(row['duration_s'] // 60)} min {int(row['duration_s'] % 60)} sec",
                max_width=300
            )
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=7,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                popup=popup
            ).add_to(fg_place_type)
        fg_place_type.add_to(m)
    
    # 7) Stops classés par catégorie (Home / Work / Autre)
    if final_stops is not None and 'place_type' in final_stops.columns:
        for category, color in [('Home','blue'), ('Work','green'), ('autre','gray')]:
            fg_cat = folium.FeatureGroup(name=f"Stops {category}", show=False)
            subset = final_stops[final_stops['place_type'] == category]
            for _, row in subset.iterrows():
                popup = folium.Popup(
                    f"<b>Lieu :</b> {row['place_type']}<br>"
                    f"<b>Début :</b> {row['start_time']}<br>"
                    f"<b>Fin :</b> {row['end_time']}<br>"
                    f"<b>Durée :</b> {int(row['duration_s']//60)} min",
                    max_width=250
                )
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=6,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.8,
                    popup=popup
                ).add_to(fg_cat)
            fg_cat.add_to(m)
    # # 7) Stops regroupés 
    # merged_close = merge_close_stops(final_stops, max_distance_m=150)

    # # créez un nouveau calque
    # fg_merged = folium.FeatureGroup(
    #     name=f"Stops fusionnés (close-stops) ({len(merged_close)})",
    #     show=True
    # )

    # for _, row in merged_close.iterrows():
    #     # rayon proportionnel à la taille du groupe
    #     radius = min(12, max(4, row['group_size'] * 2))
    #     color  = {'Home':'blue', 'Work':'green', 'autre':'gray'}[row['place_type']]
    #     popup  = folium.Popup(
    #         f"<b>Type :</b> {row['place_type']}<br>"
    #         f"<b>Début :</b> {row['start_time']}<br>"
    #         f"<b>Fin :</b> {row['end_time']}<br>"
    #         f"<b>Durée :</b> {row['duration_s']/60:.1f} min<br>"
    #         f"<b>Points fusionnés :</b> {row['group_size']}",
    #         max_width=250
    #     )
    #     folium.CircleMarker(
    #         location=(row['lat'], row['lon']),
    #         radius=radius,
    #         color=color,
    #         fill=True, fill_color=color, fill_opacity=0.8,
    #         popup=popup
    #     ).add_to(fg_merged)

    # fg_merged.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m.get_root().render()

def render_segment_report(
    df,
    stops_summary,
    grouped_stops,
    final_stops,
    evaluation,
    #matched_unknowns_df,
    figures_base64,  
    segment_start,
    segment_end
):
    """
    Retourne le HTML (string) pour un segment donné, SANS graphiques de vitesse.
    """
    html = f"""
    <hr style="margin: 40px 0;">
    <h2 id="segment_{segment_start}_{segment_end}">Segment {segment_start} → {segment_end}</h2>
    """

    # Statistiques globales du segment
    start_time = df['timestamp'].min()
    end_time   = df['timestamp'].max()
    total_recording = end_time - start_time
    nb_points = len(df)
    nb_stops  = len(stops_summary)
    total_duration_s = stops_summary['duration_s'].sum() if 'duration_s' in stops_summary.columns else 0
    total_duration_min = round(total_duration_s / 60, 1)
    total_distance_km = round(df['dist_m'].sum() / 1000, 2) if 'dist_m' in df.columns else None

    html += f"""
    <ul>
        <li><strong>Durée totale d'enregistrement :</strong> {total_recording} (du {start_time.date()} au {end_time.date()})</li>
        <li><strong>Nombre de points GPS dans le segment :</strong> {nb_points}</li>
        <li><strong>Nombre de stops détectés :</strong> {nb_stops}</li>
        <li><strong>Durée totale des stops :</strong> {total_duration_min} minutes</li>
        {f'<li><strong>Distance totale parcourue :</strong> {total_distance_km:.2f} km</li>' if total_distance_km is not None else ''}
    </ul>
    """

    # Carte interactive du segment
    map_html = generate_interactive_map(df, stops_summary, grouped_stops, final_stops)
    html += f"""
    <h3>Carte interactive</h3>
    {map_html}
    """

    # Stops détectés
    if stops_summary is not None:
        html += """
        <h3>Stops détectés</h3>
        <div class="table-container">
        """
        html += stops_summary.to_html(index=False)
        html += "</div>"

    # # Stops regroupés (bounding box)
    # if grouped_stops is not None and not grouped_stops.empty:
    #     # 1) Retirer le tz pour éviter le décalage sur to_html()
    #     df_grp = grouped_stops.copy()
    #     df_grp['start_time'] = pd.to_datetime(df_grp['start_time']).dt.tz_localize(None)
    #     df_grp['end_time']   = pd.to_datetime(df_grp['end_time']).dt.tz_localize(None)

    #     # 2) Afficher la table complète
    #     html += """
    #     <h3>Stops regroupés</h3>
    #     <div class="table-container">
    #     """
    #     html += df_grp[[
    #         'start_time','end_time','duration_s','lat','lon','group_size'
    #     ]].to_html(index=False)
    #     html += "</div>\n"

    # Lieux classifiés Home/Work/Autre
    if final_stops is not None and 'place_type' in final_stops.columns:
        html += """
        <h3>Lieux classifiés : Home / Work / Autre</h3>
        <div class="table-container">
        """
        html += final_stops[['lat','lon','place_type']].to_html(index=False)
        html += "</div>"

    # Évaluation Home/Work pour ce segment
    if evaluation:
        html += """
        <h3>Évaluation des lieux Home / Work</h3>
        <ul>
        """
        for k, v in evaluation['nombre_lieux_par_type'].items():
            html += f"<li><strong>{k}</strong> : {v} lieu(x)</li>"
        html += "</ul>\n<ul>\n"
        for k, v in evaluation['duree_cumulee_minutes'].items():
            html += f"<li><strong>{k}</strong> : {v:.1f} min</li>"
        html += "</ul>\n"
        if 'graph_hourly_distribution' in evaluation:
            html += """
            <h4>Répartition horaire des intervalles (Home/Work)</h4>
            """
            html += f"<img src=\"data:image/png;base64,{evaluation['graph_hourly_distribution']}\" width=\"700\"/><br>"

    # # Correspondance « autres » vs activités
    # if matched_unknowns_df is not None and not matched_unknowns_df.empty:
    #     html += """
    #     <h3>Correspondance des lieux "autres" avec les activités détectées</h3>
    #     <div class="table-container">
    #     """
    #     html += matched_unknowns_df[['start_time','end_time','duration_s','lat','lon','matched_activity']].to_html(index=False)
    #     html += "</div>"
    return html


def generate_full_report(
    df_all,
    stops_summary_all,
    merged_grouped_stops,
    final_stops,
    final_evaluation_merged,
    #matched_unknowns_df,
    moves_summary,
    pid=None
):
    """
    Construit la section « Résultat final » à append dans le fichier HTML.
    On y inclut :
      1) Une carte globale
      2) Le tableau des stops regroupés (avant classification finale)
      3) Le tableau des lieux classifiés finaux (après fusion close stops)
      4) L’évaluation Home/Work finale
      5) Les graphiques “Distribution des vitesses”, “Vitesse par jour/heure”, etc.
    """
    html = "<hr style=\"margin: 40px 0;\">\n"
    html += "<h2>Résultat final </h2>\n"

    # Résumé global
    html += """
    <h3>Résumé global</h3>
    <ul>
        <li><strong>Nombre total de points :</strong> {nb_points}</li>
        <li><strong>Nombre de stops détectés :</strong> {nb_stops}</li>
        <li><strong>Durée totale des stops :</strong> {durée_stops:.1f} minutes</li>
        <li><strong>Durée totale des moves :</strong> {durée_moves:.1f} minutes</li>
        <li><strong>Distance totale parcourue :</strong> {distance_km:.2f} km</li>
        <li><strong>Durée totale d'enregistrement :</strong> {total_duration} (du {start} au {end})</li>
        <li><strong>Nombre de jours couverts :</strong> {nb_jours} jour(s)</li>
        <li><strong>Jours analysés :</strong><ul>
    """.format(
        nb_points=len(df_all),
        nb_stops=len(merged_grouped_stops),
        durée_stops=merged_grouped_stops['duration_s'].sum() / 60,
        durée_moves=(df_all['timestamp'].max() - df_all['timestamp'].min()).total_seconds() / 60 - merged_grouped_stops['duration_s'].sum() / 60,
        distance_km=df_all['dist_m'].sum() / 1000,
        total_duration=df_all['timestamp'].max() - df_all['timestamp'].min(),
        start=df_all['timestamp'].min().date(),
        end=df_all['timestamp'].max().date(),
        nb_jours=len(df_all['timestamp'].dt.date.unique())
    )

    # Ajouter la liste des jours
    for day, count in df_all.groupby(df_all['timestamp'].dt.date).size().items():
        html += f"<li>{day} : {count} points</li>"
    html += "</ul></li>"

    # Statistiques vitesse et fréquence
    html += f"""
        <li><strong>Vitesse moyenne :</strong> {df_all['speed_kmh'].mean():.2f} km/h</li>
        <li><strong>Vitesse maximale :</strong> {df_all['speed_kmh'].max():.2f} km/h</li>
        <li><strong>Nombre moyen de points par jour :</strong> {int(len(df_all)/len(df_all['timestamp'].dt.date.unique()))}</li>
        <li><strong>Fréquence moyenne d’échantillonnage :</strong> un point toutes les {df_all['time_diff_s'].mean():.1f} secondes</li>
    </ul>
    """
    # juste avant de construire la carte globale
    stops_summary_all, raw_moves_all = detect_stops_and_moves(
        df_all,
        min_duration_minutes=5,
        max_diameter_meters=100,
        min_move_duration_s=30,
        min_time_gap_s=900
    )
    # stops_summary_all = detect_stops_with_skmob(
    #     df_all,
    #     epsilon_m=75,      # rayon en mètres
    #     min_time_s=15*60 
    # )

    # 1) Carte interactive globale
    html += "<h3>Carte interactive globale</h3>\n"
    map_global = generate_interactive_map(df_all, stops_summary_all, merged_grouped_stops, final_stops)
    html += map_global

    # 1bis) Stops bruts MovingPandas
    html += """
    <h3>Stops bruts détectés par scikit mobility </h3>
    <div class="table-container">
    """
    html += stops_summary_all[['start_time','end_time','duration_s','lat','lon']].to_html(index=False)
    html += "</div>\n"


    # 3) Lieux classifiés: Home / Work / Autre + DBSCAN
    html += """
    <h3>Lieux classifiés: Home / Work / Autre + DBSCAN</h3>
    <div class="table-container">
    """
    html += final_stops[['lat','lon','place_type']].to_html(index=False)
    html += "</div>\n"
    

        # ─── 1bis) Stops bruts MovingPandas → affectation place_type ───
    html += "<h3>Lieux classifiés finaux </h3>\n"

    # 1) GeoDataFrames
    gdf_raw = gpd.GeoDataFrame(
        stops_summary_all,
        geometry=gpd.points_from_xy(stops_summary_all.lon, stops_summary_all.lat),
        crs="EPSG:4326"
    )
    gdf_final = gpd.GeoDataFrame(
        final_stops,
        geometry=gpd.points_from_xy(final_stops.lon, final_stops.lat),
        crs="EPSG:4326"
    )

    # 2) Projection métrique pour distances
    gdf_raw = gdf_raw.to_crs(epsg=3857)
    gdf_final = gdf_final.to_crs(epsg=3857)

    # 3) Jointure au plus proche dans 150 m
    joined = gpd.sjoin_nearest(
        gdf_raw,
        gdf_final[['place_type','geometry']],
        how='left',
        max_distance=150,
        distance_col='distance_m'
    )

    # 4) Affichage du tableau
    html += '<div class="table-container">\n'
    html += joined[[
        'start_time','end_time','duration_s','lat','lon','place_type','distance_m'
    ]].to_html(index=False)
    html += "</div>\n"

    # … code précédent …
    joined = gpd.sjoin_nearest(
        gdf_raw,
        gdf_final[['place_type','geometry']],
        how='left',
        max_distance=150,
        distance_col='distance_m'
    )

    # On retransforme en tz-naive et on garde juste les colonnes utiles
    joined = joined.drop(columns='geometry').copy()
    joined['start_time'] = pd.to_datetime(joined['start_time']).dt.tz_localize(None)
    joined['end_time']   = pd.to_datetime(joined['end_time']).dt.tz_localize(None)


    # # 2) Stops regroupés (tous segments, avant classification)
    # html += """
    # <h3>Stops regroupés </h3>
    # <div class="table-container">
    # """
    # html += merged_grouped_stops[['start_time','end_time','duration_s','lat','lon','group_size']].to_html(index=False)
    # html += "</div>\n"

    # # 6) Correspondance des lieux "autres" avec les activités détectées
    # if matched_unknowns_df is not None and not matched_unknowns_df.empty:
    #     html += """
    #     <h3>Correspondance des lieux "autres" avec les activités </h3>
    #     <div class="table-container">
    #     """
    #     html += matched_unknowns_df[['start_time','end_time','duration_s','lat','lon','matched_activity']].to_html(index=False)
    #     html += "</div>\n"

    html += plot_rolling_speed(df_all, window_min=10)
    #html += plot_rolling_speed_with_place(df_all, final_merged_stops, window_min=10)

    # 4) Évaluation finale Home/Work
    html += """
    <h3>Évaluation finale Home / Work </h3>
    <ul>
    """
    for k, v in final_evaluation_merged['nombre_lieux_par_type'].items():
        html += f"<li><strong>{k}</strong> : {v} lieu(x)</li>\n"
    html += "</ul>\n<ul>\n"
    for k, v in final_evaluation_merged['duree_cumulee_minutes'].items():
        html += f"<li><strong>{k}</strong> : {v:.1f} min</li>\n"
    html += "</ul>\n"
    
    html += "<h3>Résumé des déplacements (Moves)</h3>\n"
    # on convertit au besoin les timestamps en tz‑naive pour éviter les décalages HTML
    df_moves = moves_summary.copy()
    if 'start_time' in df_moves.columns:
        df_moves['start_time'] = pd.to_datetime(df_moves['start_time']).dt.tz_localize(None)
    if 'end_time' in df_moves.columns:
        df_moves['end_time']   = pd.to_datetime(df_moves['end_time']).dt.tz_localize(None)

    html += "<div class=\"table-container\">\n"
    html += df_moves.to_html(index=False)
    html += "</div>\n"


    # 5) Graphiques « Vitesse » finaux (appel à generate_figures)
    # On regénère uniquement la partie “analyse de vitesse” sur le DataFrame complet
    figs = generate_figures(df_all, final_stops, None)
    #    a) Distribution des vitesses
    html += "<h3>Distribution des vitesses</h3>"
    html += f"<img src=\"data:image/png;base64,{figs['distribution_vitesse']}\" width=\"700\"/><br>"

    #    b) Distribution 0–10 km/h
    html += "<h3>Distribution des vitesses (0–10 km/h)</h3>"
    html += f"<img src=\"data:image/png;base64,{figs['distribution_vitesse_0_10_split']}\" width=\"700\"/><br>"

    #    c) Vitesses > 10 km/h
    html += "<h3>Vitesses > 10 km/h</h3>"
    html += f"<img src=\"data:image/png;base64,{figs['distribution_vitesse_haute']}\" width=\"700\"/><br>"

    #    d) Vitesse moyenne par heure
    html += "<h3>Vitesse moyenne par heure</h3>"
    html += f"<img src=\"data:image/png;base64,{figs['vitesse_par_heure']}\" width=\"700\"/><br>"

    #    e) Vitesse moyenne par jour
    html += "<h3>Vitesse moyenne par jour</h3>"
    html += f"<img src=\"data:image/png;base64,{figs['vitesse_par_jour']}\" width=\"700\"/><br>"

    #    f) Répartition hebdo vs weekend
    html += "<h3>Répartition des vitesses moyennes – Semaine vs Weekend</h3>"
    html += f"<img src=\"data:image/png;base64,{figs['vitesse_hebdo_horaire']}\" width=\"700\"/><br>"

    #    g) Heatmap Fréquence GPS – Date × Heure
    html += "<h3>Heatmap Fréquence GPS – Date × Heure</h3>"
    html += f"<img src=\"data:image/png;base64,{figs['heatmap_date_hour']}\" width=\"900\"/><br>"

    #    h) Heatmap Vitesses moyennes – Date × Heure
    html += "<h3>Heatmap des vitesses moyennes – Date × Heure</h3>"
    html += f"<img src=\"data:image/png;base64,{figs['heatmap_vitesse_date_hour']}\" width=\"900\"/><br>"

    # 6) Fin du fichier HTML
    html += "</body>\n</html>"
    return html
