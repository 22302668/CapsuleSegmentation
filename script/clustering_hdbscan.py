import matplotlib.pyplot as plt
import numpy as np
import hdbscan
from sklearn import metrics
from io import BytesIO
import base64

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def cluster_and_visualize(df):
    # Préparation des données (lat, lon)
    data_clustering = df[['lat', 'lon']].dropna().values

    silhouette_scores = []
    davies_scores = []
    calinski_scores = []

    for n in range(2, 20):
        clustering = hdbscan.HDBSCAN(min_samples=n)
        labels = clustering.fit_predict(data_clustering)

        if len(set(labels)) > 1:
            silh = metrics.silhouette_score(data_clustering, labels, metric='euclidean')
            dav = metrics.davies_bouldin_score(data_clustering, labels)
            cali = metrics.calinski_harabasz_score(data_clustering, labels)
        else:
            silh, dav, cali = -1, float('inf'), -1

        silhouette_scores.append(silh)
        davies_scores.append(dav)
        calinski_scores.append(cali)

    # Trouver le meilleur min_samples pour chaque score
    best_silhouette_idx = np.argmax(silhouette_scores) + 2
    best_davies_idx = np.argmin(davies_scores) + 2
    best_calinski_idx = np.argmax(calinski_scores) + 2

    # Appliquer HDBSCAN avec meilleur Silhouette par défaut
    final_clusterer = hdbscan.HDBSCAN(min_samples=best_silhouette_idx)
    df['cluster_behavior'] = final_clusterer.fit_predict(data_clustering)

    # Création des figures avec annotations
    fig, axs = plt.subplots(3, 1, figsize=(8, 14))

    # 1. Silhouette
    axs[0].plot(range(2, 20), silhouette_scores, marker='o')
    axs[0].set_title('Silhouette Score vs min_samples')
    axs[0].annotate(
        f"Best: {best_silhouette_idx}",
        xy=(best_silhouette_idx, silhouette_scores[best_silhouette_idx-2]),
        xytext=(best_silhouette_idx+1, silhouette_scores[best_silhouette_idx-2]+0.05),
        arrowprops=dict(arrowstyle="->", color='red'),
        fontsize=10,
        color='red'
    )

    # 2. Davies-Bouldin
    axs[1].plot(range(2, 20), davies_scores, marker='o')
    axs[1].set_title('Davies-Bouldin Score vs min_samples')
    axs[1].annotate(
        f"Best: {best_davies_idx}",
        xy=(best_davies_idx, davies_scores[best_davies_idx-2]),
        xytext=(best_davies_idx+1, davies_scores[best_davies_idx-2]+0.05),
        arrowprops=dict(arrowstyle="->", color='green'),
        fontsize=10,
        color='green'
    )

    # 3. Calinski-Harabasz
    axs[2].plot(range(2, 20), calinski_scores, marker='o')
    axs[2].set_title('Calinski-Harabasz Score vs min_samples')
    axs[2].annotate(
        f"Best: {best_calinski_idx}",
        xy=(best_calinski_idx, calinski_scores[best_calinski_idx-2]),
        xytext=(best_calinski_idx+1, calinski_scores[best_calinski_idx-2]*1.1),
        arrowprops=dict(arrowstyle="->", color='blue'),
        fontsize=10,
        color='blue'
    )

    plt.tight_layout()

    figures_base64 = {}
    figures_base64['scores_hdbscan'] = fig_to_base64(fig)

    plt.close('all')

    return df, figures_base64
