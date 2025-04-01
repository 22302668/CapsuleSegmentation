package graph;

import java.util.LinkedList;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Polygon;

public class Node implements Comparable<Node> {
   
   public static int maxId = 0;
   
   private int id;
   private Feature feature;
   private LinkedList<Edge> incidentEdges;

   public Node() {
      maxId++;
      id = maxId;
      incidentEdges = new LinkedList<Edge>(); 
   }
   
   public Feature getFeature() {
      return feature;
   }
   
   public Node(Feature f) {
      maxId++;
      id = maxId;
     
      feature = f;
      
      incidentEdges = new LinkedList<Edge>(); 
   }
   
   /**
    * return an edge between this node and node u (null if no such edge exists)
    * @param u
    * @return
    */
   public Edge getEdge(Node u) {
      for (Edge e : incidentEdges) {
         if (e.getSource() == u || e.getTarget() == u) {
            return e;
         }
      }
      return null;
   }
   
  
   
   public LinkedList<Edge> getIncidentEdges() {
      return incidentEdges;
   }
   
   public int getID() {
      return id;
   }
   
   public Coordinate getCoord() {
      return feature.getCoord();
   }
   
   
   public boolean equals(Node arg0) {
      return this.getCoord().x == arg0.getCoord().x && this.getCoord().y == arg0.getCoord().y;
   }
   
   @Override
   public int compareTo(Node arg0) {
    if (this.getCoord().x < arg0.getCoord().x) {
         return -1;
      } else if (this.getCoord().x == arg0.getCoord().x) {
         if (this.getCoord().y < arg0.getCoord().y) {
            return -1;
         } else if (this.getCoord().y == arg0.getCoord().y) {
            return 0;
         }
      } 
      return 1;
   }
   
   public String toString() {
      return id + " " + this.getCoord().x + " " + this.getCoord().y;
   }


   public void addEdge(Edge e) {
      incidentEdges.add(e);
   }

   public void setVoronoiCell(Polygon p) {
      feature.setVoronoiCell(p);
      
   }
}
