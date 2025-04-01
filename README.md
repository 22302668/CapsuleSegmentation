# GPS Location Privacy Project

This project provides tools to anonymize GPS location data of participants by truncating trajectories near sensitive areas such as homes.

## Project Structure

The project consists of two main components:
- **Mobispace**: Contains Jupyter notebooks for data preparation and analysis
- **my-home-is-my-secret**: Java implementation for geometric clustering and trajectory truncation (based on the original work by Wiedergold et al.)

## Requirements

- Python 3.6+ with libraries: pandas, geopandas, movingpandas, shapely, matplotlib
- Java JDK 8+
- Shapefiles (.shp) for centroids and trajectories

## Mobispace Data Preparation

The Mobispace directory contains Jupyter notebooks for preparing data:

### 1. `centroids.ipynb`
Extracts building locations from source data and converts them to shapefiles:
- Reads building data from CSV files
- Extracts latitude and longitude coordinates
- Creates and exports shapefiles for use with the Java components
You have to adapt the code for the head of your csv file.
You have to use the path of your file in centroid_versailles variable.

### 2. `privacy_versailles.ipynb`
Implements stop-and-move algorithms to identify trajectories requiring truncation:
- Processes GPS data from participants
- Identifies stopping points and movement segments
- Prepares trajectory data for the truncation process
You have to use the path of your file in gps_log_df variable.

### 3. `detect_stops_moves.ipynb`
Advanced implementation for detecting stops and moves in GPS trajectories:
- Uses time-based and distance-based clustering
- Identifies significant stopping locations
- Segments trajectories into stop and move episodes

### 4. `clean_trajectories.ipynb`
Preprocessing and cleaning of GPS trajectory data:
- Removes outliers and noise
- Handles missing data points
- Normalizes trajectory formats for further processing

### 5. Additional Notebooks
- `stat_versailles.ipynb`: Statistical analysis of location data
- `trying_stat.ipynb`: Experimental statistical methods

## my-home-is-my-secret Components

This component contains Java implementations for the privacy protection algorithms, originally developed by Wiedergold, Touya, and Gould.

### GeometricClustering

The GeometricClustering module creates clusters of sensitive locations (like homes):
1. Takes centroids (points of interest) as input
2. Performs Delaunay triangulation to create a graph
3. Applies geometric clustering to identify regions
4. Exports clusters as multipoints, edges, convex hulls, and Voronoi cells

To configure:
- Open `my-home-is-my-secret/my-home-is-my-secret-master/GeometricClustering/src/main/Main.java`
- Modify the input path for centroids: `path + File.separator + "input" + File.separator + "input" + File.separator + "centroids_test.shp"`

### TrajectoryTruncation

The TrajectoryTruncation module truncates trajectories near sensitive areas:
1. Takes GPS trajectories and clusters as input
2. Identifies trajectory segments entering/exiting sensitive areas
3. Truncates these segments based on geometric criteria
4. Exports the privacy-protected trajectories

To configure:
- Open `my-home-is-my-secret/my-home-is-my-secret-master/TrajectoryTruncation/src/main/Main.java`
- Modify the input paths:
  - For GPS tracks: `path + File.separator + "input" + File.separator + "input" + File.separator + "synthetic_trajectories_hel.shp"`
  - For cluster cells: `path + File.separator + "input" + File.separator + "cells.shp"`
  - For cluster multipoints: `path + File.separator + "input" + File.separator + "multipoints.shp"`

## Usage Workflow

1. **Data Preparation**:
   - Use `centroids.ipynb` to convert your centroid CSV data to shapefile format
   - Use `detect_stops_moves.ipynb` and `clean_trajectories.ipynb` for preprocessing trajectory data
   - Use `privacy_versailles.ipynb` to process trajectory data and identify segments for truncation

2. **Geometric Clustering**:
   - Update the input path in `GeometricClustering/src/main/Main.java`
   - Run the GeometricClustering application
   - This generates cluster files in the output directory

3. **Trajectory Truncation**:
   - Update the input paths in `TrajectoryTruncation/src/main/Main.java`
   - Run the TrajectoryTruncation application
   - This generates anonymized trajectory files

## Parameters

### GeometricClustering
- `k`: Number of clusters (default: 4)
- `path`: Input/output directory path

### TrajectoryTruncation
- `beta`: Angle for trajectory triangles in degrees (default: 30.0)
- `r`: Side length of triangle for visualization (default: 100.0)
- `path`: Input/output directory path

## Output Files

- `multipoints.shp`: Cluster points in each region
- `clusteredges.shp`: Edges between clusters
- `graphedges.shp`: Edges of the original graph
- `hulls.shp`: Convex hulls of clusters
- `cells.shp`: Voronoi cells representing cluster regions
- Truncated trajectory files

## References

The my-home-is-my-secret component is based on the work:
- Anna Brauer, Ville Mäkinen, Axel Forsch, Juha Oksanen & Jan-Henrik Haunert (2022) My home is my secret: concealing sensitive locations by context-aware trajectory truncation, International Journal of Geographical Information Science, 36:12, 2496-2524

Link : https://doi.org/10.1080/13658816.2022.2081694


## License

The project is licensed under the license included in the LICENSE file in the my-home-is-my-secret-master directory.
