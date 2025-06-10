
"""
S-TT Python implementation
based on "My home is my secret: concealing sensitive locations by context-aware trajectory truncation"

"""

import math
import fiona
from datetime import datetime
import os
import pandas as pd
from shapely.geometry import LineString, shape, Point, Polygon
from shapely.ops import transform
import pyproj
from rtree import index
import skmob as sm
from skmob.preprocessing import detection
import geopandas as gpd

# Version RELATIVE correcte depuis le script Python
_INPUTPATH = os.path.join(os.path.dirname(__file__), "stt_input/")
_INDEXPATH = os.path.join(os.path.dirname(__file__), "stt_index/")

def _crs_transform(shp, old, new):
    project = pyproj.Transformer.from_crs(pyproj.CRS(old), pyproj.CRS(new), always_xy=True).transform
    return transform(project, shp)


def _get_direction_between(p1, p2):
    arctangent = math.atan2(p1.y - p2.y, p1.x - p2.x)
    if arctangent < 0: arctangent += 2 * math.pi
    return math.degrees(arctangent)


def _get_endpoints(trajectory):
    return trajectory.iloc[[0, -1], 0:3].to_numpy()


def _get_stops(trajectory):  # ST-DBSCAN
    return detection.stops(trajectory, minutes_for_a_stop=15.0, spatial_radius_km=0.2).iloc[:, [0, 1, 2, 4]]\
        .to_numpy()


class STT:
    """
    This class is used to configure the S-TT algorithm
    """
    def __init__(self, pcells_crs, trajectory_crs,
                 alpha=60, k=4, buffer=0,
                 sensitive_locations=[],
                 add_endpoints=True, add_stops=False, truncation_region=None):
        """
        Initialize the S-TT object.

        :param pcells_crs: Coordinate reference system of the protection cells as a string. E.g., 'EPSG:3067'
        :param trajectory_crs: Coordinate reference system of the trajectory as a string. E.g., 'EPSG:3067'
        :param alpha: Opening angle alpha
        :param k: Clustering parameter k. This code does not execute the clustering, but it uses the parameter to
            access the right file.
        :param buffer: Size of the uncertainty buffer
        :param pbuffer: Size of the buffer region around the protection cell where all trajectory points
            are deleted as well
        :param sensitive_locations: List of sensitive locations to be entered manually. Not yet implemented.
        :param add_endpoints: If true, the trajectory's endpoints are considered sensitive locations and protected by
            S-TT.
        :param add_stops: If true, stops (i.e., stay points) in the trajectory are detected and considered sensitive
            locations
        :param truncation_region: Polygon of the region where truncation is executed. Sensitive locations outside of
            this region are ignored. Usually the extent of the set of sites.
        """

        self.sensitive_locations = sensitive_locations  # point array
        # for manually inserting sensitive locations
        # they are global, sensitive for all trajectories truncated with the STT object
        # to do
        self.alpha = alpha
        self.k = k
        self.buffer = buffer
        self.add_stops = add_stops
        self.add_endpoints = add_endpoints
        self.pcells_crs = pcells_crs
        self.trajectory_crs = trajectory_crs
        self.multipoints = self._load_multipoints()
        self.pcell_idx = index.Index(_INDEXPATH + str(self.k))
        self.truncation_region = truncation_region

    def _load_multipoints(self):
        multipoints = {}
        for feat in fiona.open(_INPUTPATH + "multipoints_" + str(self.k) + ".shp"):
            multipoints[feat['properties']['myid']] = feat['geometry']
        return multipoints

    def _get_pcells_containing_shape(self, shp):
        pcells = {}  # list that will be filled with protection cells

        # identify the protection cell(s) intersecting shp
        pc_candidates = list(self.pcell_idx.intersection(shp.bounds, objects=True))
        # are they all intersecting?
        for pc in pc_candidates:
            polygon = pc.object
            if polygon.intersects(shp):  # true intersection
                pcells[pc.id] = pc.object

        return pcells

    def _evaluate_direction(self, curr_point, prev_point, pcells):
        no_point = True
        all_points = True

        d_trajectory = _get_direction_between(curr_point, prev_point)
        max_diff = self.alpha/2

        for pcell in pcells:
            multipoints = self.multipoints[str(pcell)]
            for p_coords in multipoints['coordinates']:
                p = Point(p_coords)
                d_p = _get_direction_between(p, curr_point)
                d_diff = abs(d_p - d_trajectory)
                if d_diff <= max_diff or 360 - d_diff <= max_diff:
                    no_point = False
                else:
                    all_points = False
                if not no_point and not all_points:
                    return False  # condition not fulfilled, truncate
        return True  # do not truncate, either all or none of the points are in the wedge

    def _execute_truncation(self, points, pcells, reverse=False):
        # points is a dataframe, lng and lat contain the original coordinates,
        # geometry a transformed Shapely point
        # paramter pcells is a dictionary where the key is the timestamp of the sensitive location in the trajectory

        # get the protection cell for the end to be truncated
        if reverse:
            s_pcells = pcells[points.iloc[0, 2]]
        else:
            s_pcells = pcells[points.iloc[len(points)-1, 2]]
        for i in range(len(points)):
            if reverse:
                if i >= len(points)-1:
                    return pd.DataFrame(columns=points.columns)  # complete truncation
                curr = points.iloc[i, :]
                prev_id = i+1
            else:
                if len(points)-i-1 == 0:
                    return pd.DataFrame(columns=points.columns)  # complete truncation
                curr = points.iloc[len(points)-i-1, :]
                prev_id = len(points)-i-2

            curr_point = curr['geometry']

            curr_cell = list(self._get_pcells_containing_shape(curr_point))
            if len(curr_cell) > 0 and curr_cell[0] in s_pcells:  # proximity condition
                continue  # truncated

            if (not reverse and i != len(points)-1) or (reverse and i != 0):
                # direction condition
                prev_point = points.iloc[prev_id]['geometry']
                if not self._evaluate_direction(curr_point, prev_point, s_pcells):
                    continue  # truncated

            # if this point is reached, truncation has stopped.
            # return the truncated trajectory
            return points.iloc[i:, :] if reverse else points.iloc[:len(points)-i, :]

        return gpd.GeoDataFrame(columns=points.columns)

    def _transform_shape(self, shp):
        from_crs = pyproj.CRS(self.trajectory_crs)
        to_crs = pyproj.CRS(self.pcells_crs)
        project = pyproj.Transformer.from_crs(from_crs, to_crs, always_xy=True).transform
        transformed = transform(project, shp)
        return transformed

    def _add_pcell(self, pcells, p, ts):
        if self.buffer > 0:
            sc = p.buffer(self.buffer)
        else:
            sc = p
        s_pcells = self._get_pcells_containing_shape(sc)  # protection cells for s
        pcells[ts] = s_pcells
        return pcells

    def _split_trajectory(self, sensitive_locations, trajectory_gdf):
        pcells = {}
        sub_trajectories = []

        for i in range(len(sensitive_locations)):
            s = sensitive_locations[i]
            if len(s) > 3:  # a staypoint
                j_start = trajectory_gdf.loc[trajectory_gdf['datetime'] == s[2]].index[0]
                j_end = trajectory_gdf.loc[trajectory_gdf['datetime'] == s[3]].index[0]
                pcells = self._add_pcell(pcells=pcells, ts=s[2], p=trajectory_gdf.loc[j_start, 'geometry'])
                pcells = self._add_pcell(pcells=pcells, ts=s[3], p=trajectory_gdf.loc[j_end, 'geometry'])
                sub_trajectories.append(trajectory_gdf.loc[:j_start, :])
                trajectory_gdf = trajectory_gdf.loc[j_end:, :]
            else:  # an endpoint
                s_transformed = self._transform_shape(Point(s[0], s[1]))
                pcells = self._add_pcell(pcells=pcells, ts=s[2], p=s_transformed)

        sub_trajectories.append(trajectory_gdf)
        sub_trajectories = [x for x in sub_trajectories if len(x) > 1]
        return pcells, sub_trajectories

    def _truncate_and_reassemble(self, trajectory_gdf, sub_trajectories, sensitive_locations, pcells):
        truncated_trajectory = gpd.GeoDataFrame(columns=trajectory_gdf.columns)
        for i in range(len(sub_trajectories)):
            t = sub_trajectories[i]
            if len(sensitive_locations) > 0 and (i > 0 or t.iloc[0, 2] == sensitive_locations[0][2]):
                trunc_result_1 = self._execute_truncation(t, pcells, reverse=True)
            else:
                trunc_result_1 = t
            if len(trunc_result_1) == 0:  # if t already was truncated completely
                continue
            elif len(sensitive_locations) > 0 and \
                    (i < len(sub_trajectories) - 1 or t.iloc[-1, 2] == sensitive_locations[-1][2]):
                trunc_result_2 = self._execute_truncation(trunc_result_1, pcells)
            else:
                trunc_result_2 = trunc_result_1

            truncated_trajectory = truncated_trajectory.append(trunc_result_2)

        return truncated_trajectory

    def truncate(self, trajectories):
        """
        Execute S-TT for a set of trajectories

        :param trajectories: list of trajectories where each trajectory is a skmob.TrajDataFrame
        :return: list of truncated trajectories
        """
        output = []

        for trajectory in trajectories:
            trajectory_gdf = gpd.GeoDataFrame(trajectory, geometry=gpd.points_from_xy(trajectory.lng, trajectory.lat))
            trajectory_gdf = trajectory_gdf.set_crs(self.trajectory_crs).to_crs(self.pcells_crs)

            sloc_candidates = self.sensitive_locations.copy()

            if self.add_stops:
                sloc_candidates.extend(_get_stops(trajectory))
            if self.add_endpoints:
                sloc_candidates.extend(_get_endpoints(trajectory))

            if self.truncation_region:
                sensitive_locations = [r for r in sloc_candidates if Point(r[0], r[1]).intersects(self.truncation_region)]
            else:
                sensitive_locations = sloc_candidates

            sensitive_locations.sort(key=lambda x: x[2])

            # split trajectory at sensitive locations
            # and get all the pcells
            pcells, sub_trajectories = self._split_trajectory(sensitive_locations, trajectory_gdf)

            # truncate the sub-trajectories
            truncated_trajectory = self._truncate_and_reassemble(trajectory_gdf, sub_trajectories, sensitive_locations, pcells)
            truncated_trajectory['datetime'] = truncated_trajectory['datetime'].apply(str)
            output.append(truncated_trajectory)

        return output


def build_rtrees():
    """
    Builds index structures of the site cluster cells. Requires setting the _INDEXPATH variable to the directory
    where the indices are stored
    """
    for k in [3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]:
        idx = index.Index(_INDEXPATH + str(k))  # build spatial index (RTree)
        for feat in fiona.open(_INPUTPATH + "cells_" + str(k) + ".shp"):
            # iterate the protection cells
            geom = shape(feat['geometry'])
            myid = feat['properties']['myid']
            idx.insert(int(myid), geom.bounds, obj=geom)
        idx.close()


def geolife_to_df(path):
    """
    Parses a Geolife trajectory from its original file and creates a trajectory dataframe
    :param path: path to the Geolife plt file
    :return: skmob.TrajDataFrame
    """
    with open(path, 'r') as f:
        for _ in range(6):
            next(f)

        df = pd.DataFrame(columns=['lon', 'lat', 'time'])
        for line in f:
            items = line.split(',')
            strg = items[5] + ' ' + items[6][:-1]
            ts = datetime.strptime(strg, '%Y-%m-%d %H:%M:%S')
            lon = float(items[1])
            lat = float(items[0])
            # Filter points outside of the area covered by EPSG:2345
            if 114.0 < lon < 120.0 and 22.14 < lat < 51.52:
                df = df.append({'lon': lon, 'lat': lat,
                                'time': ts}, ignore_index=True)
        tdf = sm.TrajDataFrame(df, latitude='lat', longitude='lon', datetime='time')
        ftdf = sm.preprocessing.filtering.filter(tdf, max_speed_kmh=150)
        return ftdf


def example():
    """
    Example function, uses S-TT to truncate a trajectory from the Geolife dataset
    """
    build_rtrees()

    # load a Geolife trajectory
    trajectory_path = "/home/user/example.plt"
    tdf = geolife_to_df(trajectory_path)

    # set parameters
    k = 4
    alpha = 60
    b = 0
    beijing_study_area = Polygon([[116.0799999999999983, 39.6799999999999997],
                                  [116.0799999999999983, 40.1799999999999997],
                                  [116.7699961999999942, 40.1799999999999997],
                                  [116.7699961999999942, 39.6799999999999997]])
    stt = STT(pcells_crs='EPSG:2345', trajectory_crs='EPSG:4326',
              truncation_region=beijing_study_area,
              k=k, alpha=alpha, buffer=b,
              add_endpoints=True,
              add_stops=True)
    truncated_trajectory = stt.truncate([tdf])[0]
    print(truncated_trajectory)

