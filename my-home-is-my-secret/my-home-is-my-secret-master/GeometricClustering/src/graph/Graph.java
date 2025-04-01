package graph;

import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;

import com.vividsolutions.jts.geom.LineString;
import com.vividsolutions.jts.geom.MultiPoint;
import com.vividsolutions.jts.geom.Polygon;
import com.vividsolutions.jump.feature.AttributeType;
import com.vividsolutions.jump.feature.BasicFeature;
import com.vividsolutions.jump.feature.FeatureCollection;
import com.vividsolutions.jump.feature.FeatureDataset;
import com.vividsolutions.jump.feature.FeatureSchema;
import com.vividsolutions.jump.io.DriverProperties;
import com.vividsolutions.jump.io.ShapefileWriter;


public class Graph {
  
   
   //nodes represent clusters of buildings
   //they are ordered by cardinality of the cluster and (secondarily) by x,y of center points
   private Collection<Node> nodes;
   private List<Edge> edges;
   private HashSet<Cluster> clusters;
   private List<Edge> clusterEdges;
   
   public Graph(Collection<Node> v, List<Edge> e) {
      nodes = v;
      edges = e;
   }
      
   public void computeClusering(int minSize) {
      clusters = new HashSet<Cluster>();
      for (Node n : nodes) {
         clusters.add(n.getFeature().getCluster());
      }
      
      clusterEdges = new LinkedList<Edge>();
      
      //iterate over edges in increasing order of edge lengths
      Collections.sort(edges);
      for (Edge e : edges) {
         //System.out.println(e.getLength());
         Cluster cluster1 = e.getSource().getFeature().getCluster();
         Cluster cluster2 = e.getTarget().getFeature().getCluster();
         if (cluster1 != cluster2 && (cluster1.size() < minSize || cluster2.size() < minSize)) {
            //System.out.println("merge");
            if (cluster1.size() < cluster2.size()) {
               cluster2.union(cluster1);
               clusters.remove(cluster1);
            } else {
               cluster1.union(cluster2);
               clusters.remove(cluster2);
            }
            clusterEdges.add(e);
         } 
      }
   }
   
   public void exportEdges(String filename) {
      LinkedList<Edge> graphEdges = new LinkedList<Edge>();
      for (Node n : nodes) {
         for (Edge e : n.getIncidentEdges()) {
            if (e.getSource().getID() < e.getTarget().getID()) {
               graphEdges.add(e);
            }
         }
      }
      
      if(filename.endsWith(".shp")) {
         ShapefileWriter shp_output = new ShapefileWriter();
         DriverProperties dpw = new DriverProperties(filename);
         try {
             FeatureSchema fs = new FeatureSchema();
             fs.addAttribute("SHAPE", AttributeType.GEOMETRY);
             LinkedList<BasicFeature> myList = new LinkedList<BasicFeature>();
             
             for (Edge e : graphEdges) {
                 BasicFeature bf = new BasicFeature(fs);
                 LineString s = e.getAsLineString();
                 bf.setGeometry(s);
                 myList.push(bf);
             }    
             
             FeatureCollection myFeatureCollection = new FeatureDataset(myList, fs);
             System.out.println("Shape written to " + filename);
             shp_output.write(myFeatureCollection, dpw);
             
         } catch(Exception ex) { 
             System.out.println("shp_write: " + ex);
         }
     }
      
   }
   
   public void exportClusterEdges(String filename) {
      if(filename.endsWith(".shp")) {
         ShapefileWriter shp_output = new ShapefileWriter();
         DriverProperties dpw = new DriverProperties(filename);
         try {
             double d = 0.0;
             FeatureSchema fs = new FeatureSchema();
             fs.addAttribute("SHAPE", AttributeType.GEOMETRY);
             LinkedList<BasicFeature> myList = new LinkedList<BasicFeature>();
             
             for (Edge e : clusterEdges) {
                 BasicFeature bf = new BasicFeature(fs);
                 LineString s = e.getAsLineString();
                 d += s.getLength();
                 bf.setGeometry(s);
                 myList.push(bf);
             }
             System.out.println("Total length of cluster edges = " + d);
             
             FeatureCollection myFeatureCollection = new FeatureDataset(myList, fs);
             System.out.println("Shape written to " + filename);
             shp_output.write(myFeatureCollection, dpw);
             
         } catch(Exception ex) { 
             System.out.println("shp_write: " + ex);
         }
     }
   }
   
   public void exportClustersAsMultipoints(String filename) {
       if(filename.endsWith(".shp")) {
          ShapefileWriter shp_output = new ShapefileWriter();
          DriverProperties dpw = new DriverProperties(filename);
          try {
              FeatureSchema fs = new FeatureSchema();
              fs.addAttribute("SHAPE", AttributeType.GEOMETRY);
              fs.addAttribute("myid", AttributeType.STRING);
              LinkedList<BasicFeature> myList = new LinkedList<BasicFeature>();
              
              System.out.println("number of clusters: " + clusters.size());
              int n = 0;
              for (Cluster u : clusters) {
                  BasicFeature bf = new BasicFeature(fs);
                  MultiPoint mp = u.getAsMultiPoint();
                  bf.setGeometry(mp);
                  bf.setAttribute("myid", "" + u.getID());
                  myList.push(bf);
                  n += mp.getNumGeometries(); 
              }
              System.out.println("number of points: " + n);
              System.out.println("number of points per cluster: " + n / (double) clusters.size());
              
              
              FeatureCollection myFeatureCollection = new FeatureDataset(myList, fs);
              System.out.println("Shape written to " + filename);
              shp_output.write(myFeatureCollection, dpw);
              
          } catch(Exception ex) { 
              System.out.println("shp_write: " + ex);
          }
      }
  }



   public void exportClustersAsConvexHulls(String filename) {
      if(filename.endsWith(".shp")) {
         ShapefileWriter shp_output = new ShapefileWriter();
         DriverProperties dpw = new DriverProperties(filename);
         try {
             FeatureSchema fs = new FeatureSchema();
             double d = 0.0;
             fs.addAttribute("SHAPE", AttributeType.GEOMETRY);
             fs.addAttribute("myid", AttributeType.STRING);
             LinkedList<BasicFeature> myList = new LinkedList<BasicFeature>();
             
             for (Cluster u : clusters) {
                 BasicFeature bf = new BasicFeature(fs);
                 Polygon chull = (Polygon) u.getAsMultiPoint().convexHull();
                 bf.setGeometry(chull);
                 bf.setAttribute("myid", "" + u.getID());
                 myList.push(bf);
                 d += chull.getLength();
             }    
             System.out.println("average length of convex hull = " + d / clusters.size());
             
             FeatureCollection myFeatureCollection = new FeatureDataset(myList, fs);
             System.out.println("Shape written to " + filename);
             shp_output.write(myFeatureCollection, dpw);
             
         } catch(Exception ex) { 
             System.out.println("shp_write: " + ex);
         }
     }
   }



   public void exportClustersAsVoronoiCells(String filename) {
      if(filename.endsWith(".shp")) {
         ShapefileWriter shp_output = new ShapefileWriter();
         DriverProperties dpw = new DriverProperties(filename);
         double d = 0.0;
         try {
             FeatureSchema fs = new FeatureSchema();
             fs.addAttribute("SHAPE", AttributeType.GEOMETRY);
             fs.addAttribute("myid", AttributeType.STRING);
             LinkedList<BasicFeature> myList = new LinkedList<BasicFeature>();
             
             for (Cluster u : clusters) {
                 BasicFeature bf = new BasicFeature(fs);
                 Polygon cell = u.getVoronoiCell();
                 bf.setGeometry(cell);
                 bf.setAttribute("myid", "" + u.getID());
                 myList.push(bf);
                 d += cell.getLength();
             }    
             System.out.println("average length of voronoi cell boundary = " + d / (double) nodes.size() ); 
             
             FeatureCollection myFeatureCollection = new FeatureDataset(myList, fs);
             System.out.println("Shape written to " + filename);
             shp_output.write(myFeatureCollection, dpw);
             
         } catch(Exception ex) { 
             System.out.println("shp_write: " + ex);
         }
     }
      
   }
   
   
}
