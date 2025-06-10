import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os

def generate_home_work_centroids_shapefile(classified_stops, output_path="stt_input", epsg=4326):
    """
    À partir de classified_stops (avec 'lat', 'lon', 'place_type'), 
    créer un shapefile des centroïdes pour le clustering STT (Home/Work).
    
    Args:
        classified_stops (pd.DataFrame): Résultat de classification Home/Work
        output_path (str): Répertoire de sortie où sauver les shapefiles
        epsg (int): Code EPSG du système de coordonnées (par défaut WGS84)
    """
    os.makedirs(output_path, exist_ok=True)

    # Étape 1 : filtrer les lieux sensibles à protéger
    filtered = classified_stops[classified_stops['place_type'].isin(['Home', 'Work'])].copy()

    # Étape 2 : (optionnel) arrondir pour éviter les doublons proches
    filtered['lat'] = filtered['lat'].round(5)
    filtered['lon'] = filtered['lon'].round(5)

    # Étape 3 : convertir en GeoDataFrame
    filtered['geometry'] = filtered.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
    gdf = gpd.GeoDataFrame(filtered, geometry='geometry', crs=f"EPSG:{epsg}")

    # Étape 4 : simplifier pour le clustering
    gdf_out = gdf[['geometry']].copy()
    gdf_out['myid'] = range(1, len(gdf_out) + 1)

    # Étape 5 : exporter en shapefile
    output_file = os.path.join(output_path, "centroids.shp")
    gdf_out.to_file(output_file, driver='ESRI Shapefile')

    return output_file
