from load_and_preprocess import load_data_and_prepare
from verify_activities import verify_activities
from generate_report import generate_full_report
from movingpandas_stop_detection import detect_stops_with_movingpandas
from detect_stops_and_analyze import generate_figures
from verify_stop_activities import verify_stop_activities

from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import pandas as pd

def main():
    df = load_data_and_prepare()

    # # 1. Charger les variables d'environnement et créer engine
    # load_dotenv()
    # url = f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DB')}"
    # engine = create_engine(url)

    # print("\nChargement des données...")
    # df = load_data_and_prepare(engine)

    print("\nDétection des stops avec MovingPandas...")
    stops_summary = detect_stops_with_movingpandas(df, min_duration_minutes=3, max_diameter_meters=100)

    print("\nGénération des figures statistiques...")
    figures_base64 = generate_figures(df, stops_summary)

    print("\nVérification des activités...")
    df, activities_crosstab = verify_activities(df, engine=None)

    print("\nVérification des activités pendant les stops...")
    stops_summary, stop_activity_crosstab = verify_stop_activities(stops_summary, engine=None)

    print("\nFusion des figures et génération du rapport...")
    generate_full_report(df, stops_summary, figures_base64)

    print("\nPipeline terminé avec succès !")

if __name__ == "__main__":
    main()
