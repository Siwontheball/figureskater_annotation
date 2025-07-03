import numpy as np
from shapely.geometry import Point, Polygon

def xyxy_to_centroid(xyxy):
    """
    Convert [x1,y1,x2,y2] to (cx, cy).
    """
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def in_poly(centroid, poly_coords):
    """
    Returns True if (cx,cy) lies inside the polygon defined by poly_coords.
    """
    return Polygon(poly_coords).contains(Point(centroid))
