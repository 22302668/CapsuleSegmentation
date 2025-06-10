package graph;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Polygon;

/**
 * Class representing one point feature among a set that is to be clustered 
 * @author haunert
 *
 */
public class Feature {
	/**
	 * the coordinates of the point
	 */
   private Coordinate centroid;
   
   /**
    * a polygon corresponding to this point, e.g., the cell of a voronoi diagram
    */
   private Polygon cell;
   
   /**
    * The cluster to which the point belongs to (initially each cluster contains only one point)
    */
   private Cluster cluster;
   
   /**
    * The id of thos point
    */
   private int id;

   public Feature(Coordinate c, int id) {
      centroid = c;
      cell = null;
      this.id = id;
      cluster = new Cluster(this);
   }
   
   public Cluster getCluster() {
      return cluster;
   }

   public Coordinate getCoord() {
      return centroid;
   }

   public Polygon getVoronoiCell() {
      return cell;
   }
   
   public void setVoronoiCell(Polygon cell) {
      this.cell = cell; 
   }

   public void setCluster(Cluster cluster2) {
      cluster = cluster2;
      
   }

   public int getID() {
      return id;
   }


}
