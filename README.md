# CapsuleSegmentation

## English

### Overview
**CapsuleSegmentation** is a Python-based framework for analyzing and segmenting GPS trajectories.  
It provides tools for:
- Detecting stops and moves from raw GPS data.
- Grouping and merging close stops.
- Classifying stops into **Home**, **Work**, or **Unknown**.
- Applying clustering methods (e.g., DBSCAN).
- Generating detailed interactive reports (HTML, plots, maps).
- Supporting both **MovingPandas** and **scikit-mobility** workflows.

This project is part of the **Capsule** suite (with [CapsuleAnonymisation](https://github.com/22302668/CapsuleAnonymisation)) and focuses on **mobility analysis and segmentation**.

---

### Main Features
- **Stop detection** using MovingPandas and custom rules.  
- **Home/Work classification** based on time and duration at locations.  
- **Stop grouping and merging** for cleaner analysis.  
- **Clustering of trajectories** (DBSCAN).  
- **Report generation** with plots, summaries, and interactive maps.  

---

### File Structure
- `main.py` → Entry point to run the full pipeline.  
- `load_and_preprocess.py` → Data loading and preprocessing.  
- `detect_stops_and_analyze.py` → Stop & move detection.  
- `merge_close_stops.py` → Merge nearby stops.  
- `classify_home_work.py` / `evaluate_home_work.py` → Classification of stops into Home/Work/Unknown.  
- `dbscan_clustering.py` → DBSCAN clustering.  
- `group_stops.py` → Grouping and refinement of stop points.  
- `generate_report.py` → HTML report generation (maps, stats, plots).  
- `movingpandas_stop_detection.py` / `scikit_mobility.py` → Alternative approaches with different libraries.  
- `split_moves_stops.py` → Distinguish between move and stop segments.  

---

### Requirements
- Python 3.10+  
- Libraries: `pandas`, `geopandas`, `movingpandas`, `scikit-mobility`, `matplotlib`, `plotly`, `folium`, `scikit-learn`  

---

### Usage
Run the full pipeline with:
```bash
python main.py
```

This will:
1. Load and preprocess GPS data.  
2. Detect stops & moves.  
3. Cluster and classify stops.  
4. Generate an interactive HTML report.  

---

### Outputs
- Interactive HTML report with:  
  - Maps of trajectories and stops.  
  - Clustering visualizations.  
  - Home/Work classification.  
  - Statistics and plots.  

- CSVs for further analysis.

---

## Français

### Aperçu
**CapsuleSegmentation** est un framework en Python pour analyser et segmenter des trajectoires GPS.  
Il permet de :
- Détecter les arrêts et déplacements à partir de données GPS brutes.  
- Regrouper et fusionner les arrêts proches.  
- Classifier les arrêts en **Domicile**, **Travail** ou **Inconnu**.  
- Appliquer du clustering (ex. DBSCAN) aux trajectoires.  
- Générer des rapports interactifs (HTML, graphiques, cartes).  
- Utiliser aussi bien **MovingPandas** que **scikit-mobility**.  

---

### Fonctionnalités principales
- **Détection des arrêts** via MovingPandas et règles personnalisées.  
- **Classification Domicile/Travail** basée sur le temps et la durée passée sur site.  
- **Regroupement et fusion des arrêts** pour simplifier l’analyse.  
- **Clustering de trajectoires** avec DBSCAN.  
- **Génération de rapports** détaillés avec graphiques et cartes interactives.  

---

### Structure des fichiers
- `main.py` → Point d’entrée pour exécuter le pipeline complet.  
- `load_and_preprocess.py` → Chargement et prétraitement des données.  
- `detect_stops_and_analyze.py` → Détection des arrêts/déplacements.  
- `merge_close_stops.py` → Fusion des arrêts proches.  
- `classify_home_work.py` / `evaluate_home_work.py` → Classification des arrêts en Domicile/Travail/Inconnu.  
- `dbscan_clustering.py` → Clustering DBSCAN.  
- `group_stops.py` → Regroupement des arrêts.  
- `generate_report.py` → Génération du rapport HTML.  
- `movingpandas_stop_detection.py` / `scikit_mobility.py` → Méthodes alternatives avec différentes bibliothèques.  
- `split_moves_stops.py` → Séparation arrêts/déplacements.  

---

### Prérequis
- Python 3.10+  
- Bibliothèques : `pandas`, `geopandas`, `movingpandas`, `scikit-mobility`, `matplotlib`, `plotly`, `folium`, `scikit-learn`  

---

### Utilisation
Exécuter le pipeline complet avec :
```
python main.py
```

Cela va :
1. Charger et préparer les données GPS.  
2. Détecter les arrêts et déplacements.  
3. Effectuer le clustering et la classification.  
4. Générer un rapport HTML interactif.  

---

### Résultats
- Rapport HTML interactif incluant :  
  - Cartes des trajectoires et arrêts.  
  - Visualisations de clustering.  
  - Classification Domicile/Travail.  
  - Statistiques et graphiques.  

- Fichiers CSV pour analyse ultérieure.  
