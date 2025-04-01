package main;

import java.awt.Color;
import java.awt.Dimension;
import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.GeometryCollection;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.geom.LineString;
import com.vividsolutions.jts.geom.MultiPoint;
import com.vividsolutions.jts.geom.Point;
import com.vividsolutions.jts.geom.Polygon;
import com.vividsolutions.jts.triangulate.DelaunayTriangulationBuilder;

import graph.Edge;
import graph.Feature;
import graph.Graph;
import graph.Node;
import mapViewer.LineMapObject;
import mapViewer.ListLayer;
import mapViewer.MapFrame;
import mapViewer.MapObject;
import mapViewer.PointMapObject;

public class Main {

	public static void main(String[] args) {

		// define parameter k for clustering (either args[0] or 4 by default)
		int k = 4;
		if (args.length > 0) {
			k = Integer.parseInt(args[0]);
		}

		// define output path (either args[1] or working directory by default)
		Path currentRelativePath = Paths.get("");
		String path = currentRelativePath.toAbsolutePath().toString(); // default
		if (args.length > 1) {
			path = args[1]; // from args
			if (path.endsWith("/"))
				path = path.substring(0, path.length() - 1);
		}

		System.out.println(path);

		MapFrame myMapFrame = new MapFrame("IGGGIS - ListLayer", true);
		myMapFrame.setPreferredSize(new Dimension(1000, 600));
		myMapFrame.pack();

		// layer 1 - points
		ListLayer l1 = ListLayer.readFromShapefile(
				path + File.separator + "input" + File.separator + "input" + File.separator +  "centroids_test.shp", Color.DARK_GRAY);
		myMapFrame.getMap().addLayer(l1, 1);

		ArrayList<Coordinate> cl = new ArrayList<Coordinate>();

		TreeSet<Node> nodes = new TreeSet<Node>();
		int id = 0;
		for (MapObject myMapObject : l1.getMyObjects()) {
			if (myMapObject instanceof PointMapObject) {
				Point p = ((PointMapObject) myMapObject).getMyPoint();
				for (Coordinate c : p.getCoordinates()) {
					cl.add(c);
					Node n = new Node(new Feature(c, id));
					nodes.add(n);
					id++;
				}
			}
		} // now all vertices have been collected in list cl

		ArrayList<Edge> edgeList = new ArrayList<Edge>();

		System.out.println("n points:" + cl.size());
		System.out.println("n nodes:" + nodes.size());

		// triangulate points (triangulation will be used as input graph for clustering)
		GeometryFactory gf = new GeometryFactory();
		MultiPoint mp = gf.createMultiPoint(cl.toArray(new Coordinate[0]));
		DelaunayTriangulationBuilder dtb = new DelaunayTriangulationBuilder();
		dtb.setSites(mp);
		GeometryCollection edges = (GeometryCollection) dtb.getEdges(gf);

		// compute voronoi cells (will be exported as polygonal represenations of
		// clusters)
		@SuppressWarnings("rawtypes") // jts QuadEdgeSubdivision only returns raw type
		List vd = dtb.getSubdivision().getVoronoiCellPolygons(gf);
		for (Object o : vd) {
			Polygon p = (Polygon) o;
			Coordinate center = (Coordinate) p.getUserData();
			Node nQuery = new Node(new Feature(center, 0));
			Node nResult = nodes.floor(nQuery);
			if (nQuery.equals(nResult)) {
				nResult.setVoronoiCell(p);
			} else {
				System.out.println("ERROR");
			}
		}

		// create layer l2 for display of edges
		ListLayer l2 = new ListLayer(Color.BLACK);

		System.out.println("n edges:" + edges.getNumGeometries());
		System.out.println("n voronoi cells:" + vd.size());

		// create edges and add them to layer l2
		for (int i = 0; i < edges.getNumGeometries(); i++) {
			LineString edge = (LineString) edges.getGeometryN(i);
			LineMapObject pmo = new LineMapObject(edge);
			l2.add(pmo);

			Coordinate source = edge.getCoordinateN(0);
			Coordinate target = edge.getCoordinateN(1);

			Node sourceNode = nodes.floor(new Node(new Feature(source, 0)));
			Node targetNode = nodes.floor(new Node(new Feature(target, 0)));

			Edge e = new Edge(sourceNode, targetNode);
			e.setLength(source.distance(target));
			edgeList.add(e);
			sourceNode.addEdge(e);
			targetNode.addEdge(e);
		}

		myMapFrame.getMap().addLayer(l2, 2);
		myMapFrame.setVisible(true);

		// clustering
		System.out.println("construct graph");
		Graph g = new Graph(nodes, edgeList);
		System.out.println("start clustering");
		g.computeClusering(k);
		System.out.println("stop clustering");

		System.out.println("n nodes:" + nodes.size());

		// export results
		new File(path + File.separator + "output").mkdir();
		g.exportClustersAsMultipoints(path + File.separator + "input" + File.separator + "multipoints.shp");
		g.exportClusterEdges(path + File.separator + "input" + File.separator + "clusteredges.shp");
		g.exportEdges(path + File.separator + "input" + File.separator + "graphedges.shp");
		g.exportClustersAsConvexHulls(path + File.separator + "input" + File.separator + "hulls.shp");
		g.exportClustersAsVoronoiCells(path + File.separator + "input" + File.separator + "cells.shp");

	}
}
