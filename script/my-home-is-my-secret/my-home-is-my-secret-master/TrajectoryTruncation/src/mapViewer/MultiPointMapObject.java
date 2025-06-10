package mapViewer;

import java.awt.Graphics2D;

import com.vividsolutions.jts.geom.Envelope;
import com.vividsolutions.jts.geom.MultiPoint;
import com.vividsolutions.jts.geom.Point;

public class MultiPointMapObject implements MapObject {
   private MultiPoint mp;
   private Integer id;
   
   public MultiPointMapObject(MultiPoint mp) {
      this.mp = mp;
   }

   @Override
   public void draw(Graphics2D g, Transformation t) {
      for (int i = 0; i < mp.getNumGeometries(); i++) {
         Point myPoint = (Point) mp.getGeometryN(i);
         g.fillOval(t.getColumn(myPoint.getX()) - 2, t.getRow(myPoint.getY()) - 2, 5, 5);
      }
  }

   @Override
   public Envelope getBoundingBox() {
      return mp.getEnvelopeInternal();
   }

   public Integer getId() {
      return id;
   }

   public void setId(int id2) {
      this.id = id2;
   }

   public MultiPoint getMultiPoint() {
      return mp;
   }

}
