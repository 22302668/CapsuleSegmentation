package mapViewer;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Envelope;
import com.vividsolutions.jts.geom.LineString;
import com.vividsolutions.jts.geom.Point;


public class LineMapObject implements MapObject {

	private LineString myLineString;
	private String myName; //name des Strasse
	
	public String getMyName() {
		return myName;
	}
	
	public LineString getMyLineString() {
		return myLineString;
	}

	public LineMapObject(LineString ls) {
		this.myLineString = ls;
		this.myName = "";
	}
	
	public LineMapObject(String name, LineString ls) {
		this.myName = name;
		this.myLineString = ls;
	}
		
	@Override
	public void draw(Graphics2D g, Transformation t) {
		
		Coordinate[] coords = myLineString.getCoordinates();
		int n = coords.length;
		int[] x = new int[n];
		int[] y = new int[n];
		for(int i=0; i<n; i++) {
			x[i] = t.getColumn(coords[i].x);
			y[i] = t.getRow(coords[i].y);			
		}
		g.drawPolyline(x, y, n);
		Point centr = getMyLineString().getCentroid();
		Color oldColor = g.getColor();
		g.setColor(Color.BLACK);
		Font font = new Font("Arial", Font.PLAIN, 10);
	    g.setFont(font);
	    String name = getMyName();
	    if (name.length()>0)
	    	g.drawString(name, t.getColumn(centr.getX()), t.getRow(centr.getY()));
	    g.setColor(oldColor);
	}

	@Override
	public Envelope getBoundingBox() {

		return myLineString.getEnvelopeInternal();
	}

}
