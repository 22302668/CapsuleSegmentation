package mapViewer;


import java.awt.Graphics2D;
import com.vividsolutions.jts.geom.Envelope;
import com.vividsolutions.jts.geom.Point;

public class PointMapObject implements MapObject {

	private Point myPoint;
	
	/**
	 * @return the myPoint
	 */
	public Point getMyPoint() {
		return myPoint;
	}
	
	public PointMapObject(Point p) {
		myPoint = p;
	}

	@Override
	public void draw(Graphics2D g, Transformation t) {
		g.fillOval(t.getColumn(myPoint.getX()) - 2, t.getRow(myPoint.getY()) - 2, 5, 5);
	}

	@Override
	public Envelope getBoundingBox() {
		return new Envelope(myPoint.getX(),myPoint.getX(),myPoint.getY(),myPoint.getY());
	}

}
