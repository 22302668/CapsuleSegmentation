package graph;

import java.util.LinkedList;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.CoordinateSequence;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.geom.MultiPoint;
import com.vividsolutions.jts.geom.Point;
import com.vividsolutions.jts.geom.Polygon;
import com.vividsolutions.jts.geom.impl.CoordinateArraySequence;
import com.vividsolutions.jts.operation.union.CascadedPolygonUnion;

public class Cluster {
   
   private LinkedList<Feature> features;
   private int id;
   
   public Cluster(Feature f) {
      features = new LinkedList<Feature>();
      features.add(f);
      id = f.getID();
      //System.out.println(id);
   }
   
   public LinkedList<Feature> getFeatures() {
      return features;
   }
   
   public void union(Cluster c) {
      for (Feature f : c.features) {
         f.setCluster(this);
      }
      features.addAll(c.features);
   }

   public int size() {
      return features.size();
   }

   public MultiPoint getAsMultiPoint() {
      GeometryFactory gf = new GeometryFactory();
      Point[] points = new Point[features.size()];
      int i = 0;
      for (Feature f : features) {
         Coordinate[] c = {f.getCoord()};
         CoordinateSequence cs = new CoordinateArraySequence(c);
         points[i] = new Point(cs , gf);
         i++;
      }
      MultiPoint mp = new MultiPoint(points , gf);
      return mp;
   }

   public int getID() {
      return id;
   }

   public Polygon getVoronoiCell() {
      LinkedList<Polygon> myList =  new LinkedList<Polygon>();
      for (Feature f : features) {
         myList.add(f.getVoronoiCell());
      }
      CascadedPolygonUnion cpa = new CascadedPolygonUnion(myList);
      Polygon p = (Polygon) cpa.union();
      return p;
   }  
}
