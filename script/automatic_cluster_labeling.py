from sklearn.cluster import KMeans
import numpy as np

def assign_cluster_labels_kmeans(df, n_clusters_behavior=4):
    """
    Attribue des labels ('Arrêt', 'Marche', 'Vélo', 'Transport') aux clusters HDBSCAN
    en utilisant KMeans sur la vitesse moyenne par cluster.

    Args:
        df (pd.DataFrame): DataFrame contenant 'cluster_behavior' et 'speed_kmh_smooth'.
        n_clusters_behavior (int): Nombre de classes comportementales à détecter.

    Returns:
        df (pd.DataFrame): DataFrame enrichi avec la colonne 'cluster_label'.
    """
    # Calcul vitesse moyenne par cluster
    cluster_speed = df.groupby('cluster_behavior')['speed_kmh_smooth'].mean().dropna()

    # Filtrer les clusters existants (-1 = bruit ignoré)
    valid_clusters = cluster_speed[cluster_speed.index != -1]

    if len(valid_clusters) == 0:
        print("⚠️ Aucun cluster valide détecté.")
        df['cluster_label'] = 'Unknown'
        return df

    speeds = valid_clusters.values.reshape(-1, 1)

    # Appliquer KMeans sur les vitesses moyennes
    n_clusters_behavior = min(n_clusters_behavior, len(valid_clusters))
    kmeans = KMeans(n_clusters=n_clusters_behavior, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(speeds)

    # Associer cluster HDBSCAN ➔ cluster comportemental
    kmeans_mapping = {cluster: label for cluster, label in zip(valid_clusters.index, kmeans_labels)}

    # Maintenant, donner des noms explicites en fonction des vitesses des groupes
    cluster_mean_speed_by_behavior = {label: speeds[kmeans_labels == label].mean() for label in range(n_clusters_behavior)}
    sorted_behavior = sorted(cluster_mean_speed_by_behavior.items(), key=lambda x: x[1])

    behavior_names = ['Arrêt', 'Marche', 'Vélo', 'Transport']
    behavior_mapping = {label: behavior_names[i] for i, (label, _) in enumerate(sorted_behavior)}

    def get_label(row):
        if row['cluster_behavior'] == -1:
            return 'Bruit'
        return behavior_mapping.get(kmeans_mapping.get(row['cluster_behavior'], -1), 'Unknown')

    df['cluster_label'] = df.apply(get_label, axis=1)

    return df
