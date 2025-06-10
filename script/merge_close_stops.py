from geopy.distance import geodesic
import pandas as pd

def merge_close_stops(df: pd.DataFrame, max_distance_m: float = 100) -> pd.DataFrame:
    """
    Regroupe tous les arrêts (Home / Work / autre) dont la distance géographique
    est <= max_distance_m, en appliquant une fusion transitive.

    Args:
        df (pd.DataFrame): Doit contenir au minimum ces colonnes :
            - 'lat'           (float)
            - 'lon'           (float)
            - 'place_type'    (str : "Home", "Work" ou "autre")
            - 'start_time'    (pd.Timestamp)
            - 'end_time'      (pd.Timestamp)
            - 'duration_s'    (float)
        max_distance_m (float): Distance (en mètres) à l’intérieur de laquelle
            on fusionne deux arrêts (ou transitive via un chaînage A–B + B–C).

    Returns:
        pd.DataFrame: Nouveau DataFrame dont chaque ligne est un arrêt fusionné.
        Colonnes renvoyées :
            - place_type       (str : "Home", "Work" ou "autre", avec priorité Home→Work→autre)
            - start_time       (Timestamp : début le plus petit du groupe)
            - end_time         (Timestamp : fin la plus grande du groupe)
            - duration_s       (float : somme des durations des arrêts du groupe)
            - lat              (float : latitude moyenne des arrêts fusionnés)
            - lon              (float : longitude moyenne)
            - group_size       (int : nombre d'arrêts initialement fusionnés)
            - merged_intervals (list[str] : liste de chaînes "YYYY-MM-DD HH:MM:SS" de chaque start_time d'origine)
            - merged_ends      (list[str] : idem pour les end_time d'origine)
    """
    if df is None or df.empty:
        # Retourne un DataFrame vide disposant des colonnes attendues
        return pd.DataFrame(columns=[
            'place_type', 'start_time', 'end_time', 'duration_s',
            'lat', 'lon', 'group_size', 'merged_intervals', 'merged_ends'
        ])

    # Travailler sur une copie “nettoyée”
    df_copy = df.copy().reset_index(drop=True)

    n = len(df_copy)
    visited = set()      # indices déjà fusionnés dans un groupe
    merged_groups = []   # liste de dictionnaires (une ligne fusionnée par dict)

    for i in range(n):
        if i in visited:
            continue

        # Démarrage d’un nouveau groupe avec l’élément pivot i
        pivot = df_copy.loc[i]
        current_rows = [pivot]  
        visited.add(i)

        # On cherche tout arrêt j > i qui peut s’ajouter, soit directement via pivot,
        # soit transitivement via les lignes déjà ajoutées à current_rows.
        # Tant qu’on ajoute quelque chose, on continue à itérer sur la liste “à la volée”.
        idx_to_scan = 0
        while idx_to_scan < len(current_rows):
            reference = current_rows[idx_to_scan]
            # référence = un des arrêts déjà dans current_rows
            for j in range(n):
                if j in visited:
                    continue
                candidate = df_copy.loc[j]
                # Calcul de distance entre “reference” et “candidate”
                distance_m = geodesic(
                    (reference['lat'], reference['lon']),
                    (candidate['lat'], candidate['lon'])
                ).meters
                if distance_m <= max_distance_m:
                    # On ajoute “candidate” dans le groupe
                    current_rows.append(candidate)
                    visited.add(j)
            idx_to_scan += 1

        # À ce stade, current_rows contient TOUS les arrêts fusionnés (chaînage transitive)
        temp_df = pd.DataFrame(current_rows)

        # Détermination du label final (priorité Home → Work → autre)
        if (temp_df['place_type'] == 'Home').any():
            final_label = 'Home'
        elif (temp_df['place_type'] == 'Work').any():
            final_label = 'Work'
        else:
            final_label = 'autre'

        # Construction du dictionnaire de sortie pour ce groupe
        merged_groups.append({
            'place_type'      : final_label,
            'start_time'      : temp_df['start_time'].min(),
            'end_time'        : temp_df['end_time'].max(),
            'duration_s'      : temp_df['duration_s'].sum(),
            'lat'             : temp_df['lat'].mean(),
            'lon'             : temp_df['lon'].mean(),
            'group_size'      : len(temp_df),
            'merged_starts': list(temp_df['start_time'].dt.strftime('%Y-%m-%d %H:%M:%S')),
            'merged_ends'     : list(temp_df['end_time'].dt.strftime('%Y-%m-%d %H:%M:%S'))
        })

    return pd.DataFrame(merged_groups)
