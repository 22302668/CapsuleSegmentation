package mapViewer;

import java.awt.Color;
import java.util.LinkedList;
import java.util.List;

import com.vividsolutions.jts.geom.Envelope;
import com.vividsolutions.jts.index.strtree.STRtree;

public class TreeLayer extends Layer {

	private STRtree myObjects;

	public TreeLayer(Color c) {
		super(c);
		extent = null;
    	myObjects = new STRtree();
	}

	@Override
	public List<MapObject> query(Envelope searchEnv) {
		LinkedList<MapObject> result = new LinkedList<MapObject>();
		for (Object o : myObjects.query(searchEnv)) {
			MapObject mo = (MapObject) o;
			result.add(mo);
		}
		return result;
	}

	public void add(MapObject mo) {
		myObjects.insert(mo.getBoundingBox(), mo);
    	if (extent == null) {
    		extent = mo.getBoundingBox();
    	} else {
    		extent.expandToInclude(mo.getBoundingBox());
    	}
	}

}
