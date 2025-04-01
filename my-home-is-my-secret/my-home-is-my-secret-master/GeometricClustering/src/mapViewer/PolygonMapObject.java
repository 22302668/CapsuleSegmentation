package mapViewer;

import java.awt.Graphics2D;

import com.vividsolutions.jts.geom.Coordinate;
//Download https://sourceforge.net/projects/jts-topo-suite/
import com.vividsolutions.jts.geom.Envelope;
import com.vividsolutions.jts.geom.Polygon;

public class PolygonMapObject implements MapObject {

	private Polygon polygon;

	public PolygonMapObject(Polygon polygon) {
		this.polygon = polygon;
	}

	@Override
	public void draw(Graphics2D g, Transformation t) {
		java.awt.Polygon awtPolygon = new java.awt.Polygon();
		for (Coordinate p : polygon.getCoordinates()) {
			awtPolygon.addPoint(t.getColumn(p.x), t.getRow(p.y));
		}
		g.drawPolygon(awtPolygon);
	}

	@Override
	public Envelope getBoundingBox() {
		Envelope bbox = polygon.getEnvelopeInternal();
		return bbox;
	}
}
