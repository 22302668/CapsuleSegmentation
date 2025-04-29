from load_and_preprocess import load_data_and_prepare
from clustering_hdbscan import cluster_and_visualize
from detect_stops_and_analyze import detect_stops_and_generate_figures
from verify_activities import verify_activities
from generate_report import generate_full_report
from automatic_cluster_labeling import assign_cluster_labels_kmeans

from sqlalchemy import create_engine
from dotenv import load_dotenv
import os


def main():
    # 1. Charger les variables d'environnement et créer engine une seule fois
    load_dotenv()
    url = f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DB')}"
    engine = create_engine(url)

    print("\n🚀 Chargement des données...")
    df = load_data_and_prepare(engine)

    print("\n🧠 Clustering HDBSCAN...")
    df, clustering_figures = cluster_and_visualize(df)
    df = assign_cluster_labels_kmeans(df)

    print("\n🏃 Détection des stops et analyse des mouvements...")
    df, stops_summary, stops_figures = detect_stops_and_generate_figures(df)

    print("\n🧩 Vérification des activités...")
    df, activities_crosstab = verify_activities(df, engine)

    print("\n📄 Fusion des figures et génération du rapport...")
    all_figures = {**clustering_figures, **stops_figures}
    generate_full_report(df, stops_summary, all_figures, activities_crosstab)

    print("\n✅ Pipeline terminé avec succès !")


if __name__ == "__main__":
    main()
