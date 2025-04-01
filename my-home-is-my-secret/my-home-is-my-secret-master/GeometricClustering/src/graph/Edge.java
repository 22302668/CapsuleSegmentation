package graph;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.CoordinateSequence;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.geom.LineString;
import com.vividsolutions.jts.geom.impl.CoordinateArraySequence;

public class Edge implements Comparable<Edge>{
   
   private Node source;
   private Node target;
   private double length;
   
   public Edge(Node sourceNode, Node targetNode) {
      source = sourceNode;
      target = targetNode;
   }

   public Node getSource() {
      return source;
   }

   public Node getTarget() {
      return target;
   }

   public double getLength() {
      return length;
   }

   public void setLength(double length) {
      this.length = length;
   }



   public LineString getAsLineString() {
      Coordinate[] coord = new Coordinate[2];
      coord[0] = source.getCoord();
      coord[1] = target.getCoord();
      CoordinateSequence coords = new CoordinateArraySequence(coord);
      LineString ls = new LineString(coords, new GeometryFactory());
      return ls;
   }



   @Override
   public int compareTo(Edge arg0) {
      
      //comparison based on length
      if (this.length < arg0.length) {
         return -1;
      } else if (this.length > arg0.length) {
         return 1;
      }
      
      //comparison based on id of source
      if (this.source.getID() < arg0.getSource().getID()) {
         return -1;
      } else if (this.source.getID() > arg0.getSource().getID()) {
         return 1;
      }
      
      //comparison based on id of target
      if (this.target.getID() < arg0.getTarget().getID()) {
         return -1;
      } else if (this.target.getID() > arg0.getTarget().getID()) {
         return 1;
      }
      
      return 0;
   }

  
   
  
   
}
