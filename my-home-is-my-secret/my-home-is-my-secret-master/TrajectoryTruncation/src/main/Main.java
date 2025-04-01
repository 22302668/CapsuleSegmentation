package main;

import java.awt.Color;
import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.TreeMap;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Envelope;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.geom.LineString;
import com.vividsolutions.jts.geom.MultiPoint;
import com.vividsolutions.jts.geom.Point;
import com.vividsolutions.jts.geom.Polygon;
import com.vividsolutions.jts.geom.impl.CoordinateArraySequence;
import com.vividsolutions.jts.index.strtree.STRtree;
import com.vividsolutions.jump.feature.AttributeType;
import com.vividsolutions.jump.feature.BasicFeature;
import com.vividsolutions.jump.feature.FeatureCollection;
import com.vividsolutions.jump.feature.FeatureDataset;
import com.vividsolutions.jump.feature.FeatureSchema;
import com.vividsolutions.jump.io.DriverProperties;
import com.vividsolutions.jump.io.ShapefileWriter;

import mapViewer.LineMapObject;
import mapViewer.ListLayer;
import mapViewer.MapObject;
import mapViewer.MultiPointMapObject;
import mapViewer.PolygonMapObject;

public class Main {

	// beta = alpha/2; alpha = angle of triangle in last point of track
	private static double beta = 30.0;

	// r = side length (leg) of triangle for visualization
	// for testing point within triangle, length of triangle leg is assumed infinity
	private static double r = 100.0;

	public static void main(String[] args) {

		// define parameter beta for triangles
		if (args.length > 0) {
			beta = Double.parseDouble(args[0]);
		}

		// define parameter r for triangles
		if (args.length > 1) {
			r = Double.parseDouble(args[1]);
		}

		// input+output path
		Path currentRelativePath = Paths.get("");
		String path = currentRelativePath.toAbsolutePath().toString(); // default
		if (args.length > 2) {
			path = args[2]; // from args
			if (path.endsWith("/"))
				path = path.substring(0, path.length() - 1);
		}

		// shp file with GPS tracks
		ListLayer tracksList = ListLayer.readFromShapefile(
				path + File.separator + "input" + File.separator + "input" + File.separator + "synthetic_trajectories_hel.shp", Color.DARK_GRAY);

		// shp files with a polygons corresponding to the merged voronoi cells for each
		// cluster
		ListLayer cellsList = ListLayer
				.readFromShapefile(path + File.separator + "input" + File.separator + "cells.shp", Color.DARK_GRAY);

		// shp file with a multipoint for each cluster
		ListLayer clustersList = ListLayer.readFromShapefile(
				path + File.separator + "input" + File.separator + "multipoints.shp", Color.DARK_GRAY);

		GeometryFactory gf = new GeometryFactory();

		// spatial index containing, for each cluster c, the corresponding polygonal
		// region
		STRtree cellsSpatialIndex = new STRtree();
		// data structure containing, for each cluster c, the id and the corresponding
		// polygonal region
		TreeMap<Integer, PolygonMapObject> cellsTree = new TreeMap<Integer, PolygonMapObject>();
		for (MapObject o : cellsList.getMyObjects()) {
			PolygonMapObject pmo = (PolygonMapObject) o;
			cellsTree.put(pmo.getId(), pmo);
			cellsSpatialIndex.insert(pmo.getPolygon().getEnvelopeInternal(), pmo);
		}

		// data structure containing, for each cluster c, the id and the corresponding
		// multipoint
		TreeMap<Integer, MultiPointMapObject> clustersTree = new TreeMap<Integer, MultiPointMapObject>();
		for (MapObject o : clustersList.getMyObjects()) {
			MultiPointMapObject pmo = (MultiPointMapObject) o;
			clustersTree.put(pmo.getId(), pmo);
		}

		// list of all tracks to be processed
		LinkedList<LineString> tracks = new LinkedList<>();
		for (MapObject o : tracksList.getMyObjects()) {
			LineMapObject ls = (LineMapObject) o;
			tracks.add(ls.getMyLineString());
		}

		LinkedList<LineString> newLineStrings = new LinkedList<>();
		LinkedList<Polygon> triangles = new LinkedList<>();
		ArrayList<Double> triangleRotations = new ArrayList<>();

		int finalStart = -1, finalEnd = -1;

		// Now truncate tracks
		for (LineString ls : tracks) {
			//
			Coordinate firstPoint = ls.getCoordinateN(0);
			Coordinate lastPoint = ls.getCoordinateN(ls.getNumPoints() - 1);

			// identify the region containing the first and last trajectory point
			PolygonMapObject firstCell = getCellContainingPoint(cellsSpatialIndex, firstPoint);
			PolygonMapObject lastCell = getCellContainingPoint(cellsSpatialIndex, lastPoint);
			MultiPointMapObject firstCluster = clustersTree.get(firstCell.getId());
			MultiPointMapObject lastCluster = clustersTree.get(lastCell.getId());

			// find first point of track to keep
			int left = 0;
			while (left < ls.getNumPoints()) {
				boolean keepPoint = true; // by default a point is kept
				Coordinate currentPoint = ls.getCoordinateN(left);
				PolygonMapObject currentCell = getCellContainingPoint(cellsSpatialIndex, currentPoint);
				if (firstCell == currentCell) {
					keepPoint = false; // point lies in region of home cluster - do not keep!
				} else if (left < ls.getNumPoints() - 1) {
					Coordinate prevPoint = ls.getCoordinateN(left + 1);
					double alpha = Math.atan2(currentPoint.y - prevPoint.y, currentPoint.x - prevPoint.x);
					if (alpha < 0) {
						alpha += 2 * Math.PI;
					}
					// construct triangle with side length r
					double alpha1 = alpha + beta * Math.PI / 180.0;
					double x1 = currentPoint.x + r * Math.cos(alpha1);
					double y1 = currentPoint.y + r * Math.sin(alpha1);

					double alpha2 = alpha - beta * Math.PI / 180.0;
					double x2 = currentPoint.x + r * Math.cos(alpha2);
					double y2 = currentPoint.y + r * Math.sin(alpha2);

					Coordinate[] triangleCoords = new Coordinate[4];
					triangleCoords[0] = currentPoint;
					triangleCoords[1] = new Coordinate(x1, y1);
					triangleCoords[2] = new Coordinate(x2, y2);
					triangleCoords[3] = currentPoint;
					Polygon triangle = gf.createPolygon(triangleCoords);
					triangles.add(triangle);
					triangleRotations.add(alpha);

					// i = number of points of the home cluster contained in the current triangle
					int i = countPointsInRange(currentPoint, firstCluster.getMultiPoint(), alpha,
							beta * Math.PI / 180.0);

					// if the current triangle contains some but not all points of the home cluster,
					// do not keep!
					if (i != 0 && i != firstCluster.getMultiPoint().getNumGeometries()) {
						keepPoint = false;
					}

				}

				// decide whether or not to keep the current point - if not, continue with next
				if (keepPoint) {
					break;
				} else {
					left++;
				}
			}
			finalStart = triangles.size() - 1;

			// this takes care of the other end of the trajectory, in the same way as before
			int right = ls.getNumPoints() - 1;
			while (right >= 0) {
				boolean keepPoint = true;
				Coordinate currentPoint = ls.getCoordinateN(right);
				PolygonMapObject currentCell = getCellContainingPoint(cellsSpatialIndex, currentPoint);
				if (lastCell == currentCell) {
					keepPoint = false;
				} else if (right > 0) {
					Coordinate prevPoint = ls.getCoordinateN(right - 1);
					double alpha = Math.atan2(currentPoint.y - prevPoint.y, currentPoint.x - prevPoint.x);
					if (alpha < 0) {
						alpha += 2 * Math.PI;
					}
					// construct triangle with side length r
					double alpha1 = alpha + beta * Math.PI / 180.0;
					double x1 = currentPoint.x + r * Math.cos(alpha1);
					double y1 = currentPoint.y + r * Math.sin(alpha1);

					double alpha2 = alpha - beta * Math.PI / 180.0;
					double x2 = currentPoint.x + r * Math.cos(alpha2);
					double y2 = currentPoint.y + r * Math.sin(alpha2);

					Coordinate[] triangleCoords = new Coordinate[4];
					triangleCoords[0] = currentPoint;
					triangleCoords[1] = new Coordinate(x1, y1);
					triangleCoords[2] = new Coordinate(x2, y2);
					triangleCoords[3] = currentPoint;
					Polygon triangle = gf.createPolygon(triangleCoords);
					triangles.add(triangle);
					triangleRotations.add(alpha);

					int i = countPointsInRange(currentPoint, lastCluster.getMultiPoint(), alpha,
							beta * Math.PI / 180.0);
					// System.out.println("points in range:" + i + " of " +
					// lastCluster.getMultiPoint().getNumGeometries());
					if (i != 0 && i != lastCluster.getMultiPoint().getNumGeometries()) {
						keepPoint = false;
					}
				}
				if (keepPoint) {
					break;
				} else {
					right--;
				}
			}
			finalEnd = triangles.size() - 1;

			// create new line string
			if (0 <= left && left < right && right < ls.getNumPoints()) {
				Coordinate[] coords = new Coordinate[right - left + 1];
				for (int i = 0; i < coords.length; i++) {
					coords[i] = ls.getCoordinateN(i + left);
				}

				LineString newls = gf.createLineString(coords);
				newLineStrings.add(newls);
			}

		}

		new File(path + File.separator + "output").mkdir();
		exportLineStrings(path + File.separator + "output" + File.separator + "truncated.shp", newLineStrings);
		exportTriangles(path + File.separator + "output" + File.separator + "triangles.shp", triangles,
				triangleRotations, finalStart, finalEnd);
	}

	private static int countPointsInRange(Coordinate currentPoint, MultiPoint multiPoint, double startAngle,
			double maxDif) {
		int counter = 0;
		for (int i = 0; i < multiPoint.getNumGeometries(); i++) {
			Point p = (Point) multiPoint.getGeometryN(i);
			double alpha = Math.atan2(p.getY() - currentPoint.y, p.getX() - currentPoint.x);
			if (alpha < 0) {
				alpha += 2 * Math.PI;
			}
			double dAlpha = Math.abs(alpha - startAngle);
			if (dAlpha <= maxDif || 2 * Math.PI - dAlpha <= maxDif) {
				counter++;
			}
		}
		return counter;
	}

	public static PolygonMapObject getCellContainingPoint(STRtree cellsSpatialIndex, Coordinate c) {
		Envelope env = new Envelope();
		env.expandToInclude(c);
		@SuppressWarnings("rawtypes") // jts STRtree only returns raw type
		List l = cellsSpatialIndex.query(env);
		for (Object o : l) {
			PolygonMapObject pmo = (PolygonMapObject) o;
			Coordinate[] carray = { c };
			CoordinateArraySequence ca = new CoordinateArraySequence(carray);
			if (pmo.getPolygon().contains(new Point(ca, new GeometryFactory()))) {
				if (pmo.getId() != 0)
					return pmo;
			}
		}
		return null;

	}

	public static void exportTriangles(String filename, LinkedList<Polygon> triangles,
			ArrayList<Double> triangleRotations, int finalStart, int finalEnd) {
		if (filename.endsWith(".shp")) {
			ShapefileWriter shp_output = new ShapefileWriter();
			DriverProperties dpw = new DriverProperties(filename);
			try {
				FeatureSchema fs = new FeatureSchema();
				fs.addAttribute("SHAPE", AttributeType.GEOMETRY);
				fs.addAttribute("isFinal", AttributeType.INTEGER);
				fs.addAttribute("direction", AttributeType.DOUBLE);
				LinkedList<BasicFeature> myList = new LinkedList<BasicFeature>();

				int n = 0;
				for (Polygon p : triangles) {
					BasicFeature bf = new BasicFeature(fs);
					bf.setGeometry(p);
					bf.setAttribute("isFinal", finalStart == n || finalEnd == n ? 1 : 0);
					bf.setAttribute("direction", triangleRotations.get(n++) * 180.0 / Math.PI);
					myList.push(bf);
				}

				FeatureCollection myFeatureCollection = new FeatureDataset(myList, fs);
				System.out.println("Shape written to " + filename);
				shp_output.write(myFeatureCollection, dpw);

			} catch (Exception ex) {
				System.out.println("shp_write: " + ex);
			}
		}
	}

	public static void exportLineStrings(String filename, LinkedList<LineString> ls) {
		if (filename.endsWith(".shp")) {
			ShapefileWriter shp_output = new ShapefileWriter();
			DriverProperties dpw = new DriverProperties(filename);
			try {
				FeatureSchema fs = new FeatureSchema();
				fs.addAttribute("SHAPE", AttributeType.GEOMETRY);
				LinkedList<BasicFeature> myList = new LinkedList<BasicFeature>();

				for (LineString l : ls) {
					BasicFeature bf = new BasicFeature(fs);
					bf.setGeometry(l);
					myList.push(bf);
				}

				FeatureCollection myFeatureCollection = new FeatureDataset(myList, fs);
				System.out.println("Shape written to " + filename);
				shp_output.write(myFeatureCollection, dpw);

			} catch (Exception ex) {
				System.out.println("shp_write: " + ex);
			}
		}
	}

}
