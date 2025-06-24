import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math
import random

from find_contour import detect_puzzle_pieces  # Changed from detect_pieces
from detect_corner_simple import find_square_corners


# Constants - SWAPPED DEFINITIONS
TAB = 'Tab'      
BLANK = 'Blank'  
FLAT = 'Flat'    

def process_piece(contour):
    """Process a piece to extract corners and contour points"""
    if len(contour.shape) == 3 and contour.shape[1] == 1:
        contour_points = contour.squeeze()
    else:
        contour_points = contour.copy()
    
    # Find corners
    corners = find_square_corners(contour)
    
    # Calculate centroid
    M = cv2.moments(contour)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centroid = np.array([cx, cy])
    else:
        centroid = np.mean(contour_points, axis=0)
    
    # Sort corners in clockwise order
    corners = sort_corners(corners, centroid)
    
    return contour_points, corners, centroid

def sort_corners(corners, centroid):
    """Sort corners in clockwise order"""
    angles = []
    for corner in corners:
        angle = math.atan2(corner[1] - centroid[1], corner[0] - centroid[0])
        angles.append(angle)
    
    # Sort corners by angle
    sorted_indices = np.argsort(angles)
    sorted_corners = corners[sorted_indices]
    
    return sorted_corners

def distance_to_line(point, line_start, line_end):
    """Calculate perpendicular distance from point to line"""
    # Line vector
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)
    
    if line_len < 1e-6:
        return np.linalg.norm(point - line_start), line_start
    
    # Unit vector along line
    line_unit = line_vec / line_len
    
    # Vector from line_start to point
    to_point = point - line_start
    
    # Projection of to_point onto line_unit
    proj_len = np.dot(to_point, line_unit)
    
    # Clamp projection to line segment
    proj_len = max(0, min(proj_len, line_len))
    
    # Calculate point on line
    proj_point = line_start + proj_len * line_unit
    
    # Calculate distance
    distance = np.linalg.norm(point - proj_point)
    
    return distance, proj_point

def extract_edge_contour(contour_points, corner1, corner2):
    """Extract contour segment between two corners"""
    # Find indices of closest points to corners
    dists1 = np.linalg.norm(contour_points - corner1, axis=1)
    dists2 = np.linalg.norm(contour_points - corner2, axis=1)
    
    idx1 = np.argmin(dists1)
    idx2 = np.argmin(dists2)
    
    # Edge length
    edge_len = np.linalg.norm(corner2 - corner1)
    
    # Extract segment
    if abs(idx1 - idx2) < len(contour_points) / 2:
        # Take the shorter path
        if idx1 <= idx2:
            segment = contour_points[idx1:idx2+1]
        else:
            segment = contour_points[idx2:idx1+1][::-1]
    else:
        # Take the longer path (wrap around)
        if idx1 <= idx2:
            segment = np.vstack([contour_points[idx2:], contour_points[:idx1+1]])[::-1]
        else:
            segment = np.vstack([contour_points[idx1:], contour_points[:idx2+1]])
    
    return segment, edge_len

def is_point_outward(point, midpoint, centroid):
    """Determine if a point is outward or inward relative to the piece center"""
    # Vector from centroid to midpoint
    centroid_to_mid = midpoint - centroid
    
    # Vector from midpoint to point
    mid_to_point = point - midpoint
    
    # Dot product will be negative if vectors are in opposite directions
    # (meaning point is outward)
    dot_product = np.dot(centroid_to_mid, mid_to_point)
    
    # Return True if point is outward (away from centroid)
    return dot_product < 0

def classify_edge(contour_points, corner1, corner2, centroid):
    """Classify edge type using maximum deviation approach"""
    # Extract edge contour
    edge_segment, edge_len = extract_edge_contour(contour_points, corner1, corner2)
    
    if len(edge_segment) < 5 or edge_len < 1e-6:
        return FLAT, 1.0, None, None
    
    # Calculate edge midpoint
    midpoint = (corner1 + corner2) / 2
    
    # Find point with maximum distance from line
    max_dist = 0
    max_point = None
    max_proj_point = None
    
    for point in edge_segment:
        dist, proj_point = distance_to_line(point, corner1, corner2)
        if dist > max_dist:
            max_dist = dist
            max_point = point
            max_proj_point = proj_point
    
    # If no significant deviation found
    if max_dist < 0.05 * edge_len:
        return FLAT, 1.0, max_point, max_proj_point
    
    # Determine if maximum deviation is outward or inward
    outward = is_point_outward(max_point, max_proj_point, centroid)
    
    # Calculate relative deviation
    rel_deviation = max_dist / edge_len
    
    # Classify based on direction and magnitude
    if rel_deviation < 0.1:  # Small deviation
        return FLAT, 1.0, max_point, max_proj_point
    # *** SWAPPED THE LABELS HERE ***
    elif outward:  # Outward deviation = Blank (swapped with Tab)
        return BLANK, min(1.0, rel_deviation / 0.3), max_point, max_proj_point
    else:  # Inward deviation = Tab (swapped with Blank)
        return TAB, min(1.0, rel_deviation / 0.3), max_point, max_proj_point

def analyze_piece(contour):
    """Analyze all edges of a puzzle piece"""
    contour_points, corners, centroid = process_piece(contour)
    
    edge_types = []
    confidences = []
    max_points = []
    proj_points = []
    
    for i in range(len(corners)):
        corner1 = corners[i]
        corner2 = corners[(i + 1) % len(corners)]
        
        edge_type, confidence, max_point, proj_point = classify_edge(
            contour_points, corner1, corner2, centroid)
        
        edge_types.append(edge_type)
        confidences.append(confidence)
        
        if max_point is not None:
            max_points.append(max_point)
        if proj_point is not None:
            proj_points.append(proj_point)
    
    return {
        'edge_types': edge_types,
        'confidences': confidences,
        'corners': corners,
        'centroid': centroid,
        'max_points': max_points,
        'proj_points': proj_points
    }

def classify_all_pieces(contours):
    """Classify all pieces in the image"""
    all_results = []
    
    for i, contour in enumerate(contours):
        print(f"Processing piece {i+1}/{len(contours)}...")
        
        result = analyze_piece(contour)
        result['index'] = i
        result['contour'] = contour
        
        all_results.append(result)
    
    return all_results

def create_visualization(img, results):
    """Create visualization of edge classifications"""
    output = img.copy()
    
    # Color mapping for edge types
    colors = {
        TAB: (0, 255, 0),     # Green for tabs
        BLANK: (0, 0, 255),   # Red for blanks
        FLAT: (255, 0, 0)     # Blue for flat edges
    }
    
    for result in results:
        piece_idx = result['index']
        contour = result['contour']
        corners = result['corners']
        centroid = result['centroid']
        edge_types = result['edge_types']
        
        # Draw contour
        cv2.drawContours(output, [contour], -1, (255, 255, 255), 2)
        
        # Draw piece number
        cv2.putText(output, str(piece_idx), tuple(centroid.astype(int)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Draw corners
        for i, corner in enumerate(corners):
            cv2.circle(output, tuple(corner.astype(int)), 5, (0, 255, 255), -1)
            cv2.putText(output, str(i), tuple(corner.astype(int)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw edge classifications
        for i in range(len(corners)):
            edge_type = edge_types[i]
            color = colors[edge_type]
            
            # Draw line between corners
            corner1 = corners[i]
            corner2 = corners[(i + 1) % len(corners)]
            cv2.line(output, tuple(corner1.astype(int)), tuple(corner2.astype(int)), 
                    color, 2)
            
            # Add edge type label
            midpoint = ((corner1 + corner2) / 2).astype(int)
            cv2.putText(output, edge_type, tuple(midpoint), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw maximum deviation points
        if 'max_points' in result and 'proj_points' in result:
            for point, proj in zip(result['max_points'], result['proj_points']):
                if point is not None and proj is not None:
                    # Draw point with maximum deviation
                    cv2.circle(output, tuple(point.astype(int)), 5, (255, 255, 0), -1)  # Yellow
                    
                    # Draw projection on line
                    cv2.circle(output, tuple(proj.astype(int)), 3, (0, 255, 255), -1)  # Cyan
                    
                    # Draw line connecting them
                    cv2.line(output, tuple(point.astype(int)), tuple(proj.astype(int)),
                            (0, 255, 255), 1)  # Cyan line
    
    return output

def get_canonical_signature(edge_types):
    """Get canonical signature for piece shape (rotation invariant)"""
    # Generate all rotations
    rotations = []
    for i in range(len(edge_types)):
        rotation = edge_types[i:] + edge_types[:i]
        rotations.append(tuple(rotation))
    
    # Return lexicographically smallest rotation
    return min(rotations)

def group_pieces_by_shape(results):
    """Group pieces by their canonical shape signature"""
    groups = defaultdict(list)
    
    for result in results:
        piece_idx = result['index']
        edge_types = result['edge_types']
        
        # Get canonical signature
        signature = get_canonical_signature(edge_types)
        
        # Add to group
        groups[signature].append(piece_idx)
    
    return groups

def create_shape_visualization(img, results, groups):
    """Create visualization with pieces colored by shape group"""
    output = img.copy()
    
    # Create a random color for each unique shape group
    shape_colors = {}
    for signature in groups.keys():
        # Generate a random bright color (avoid dark colors)
        r = random.randint(100, 255)
        g = random.randint(100, 255)
        b = random.randint(100, 255)
        shape_colors[signature] = (b, g, r)  # OpenCV uses BGR
    
    # Draw each piece filled with its shape group color
    for result in results:
        piece_idx = result['index']
        contour = result['contour']
        edge_types = result['edge_types']
        centroid = result['centroid']
        
        # Get canonical signature for this piece
        signature = get_canonical_signature(edge_types)
        
        # Get color for this shape group
        color = shape_colors[signature]
        
        # Fill contour with shape color
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        output[mask == 255] = color
        
        # Also draw contour outline
        cv2.drawContours(output, [contour], -1, (255, 255, 255), 2)
        
        # Draw piece index
        cv2.putText(output, str(piece_idx), tuple(centroid.astype(int)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    
    # Create a legend for the shape groups
    legend_img = np.ones((150, 300, 3), dtype=np.uint8) * 255
    y_offset = 30
    
    cv2.putText(legend_img, "Shape Groups:", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    y_offset += 30
    
    for signature, indices in groups.items():
        color = shape_colors[signature]
        
        # Convert signature to simple form for display
        simple_sig = []
        for edge in signature:
            if edge == TAB:
                simple_sig.append('T')
            elif edge == BLANK:
                simple_sig.append('B')
            else:
                simple_sig.append('F')
        
        # Draw color rectangle
        cv2.rectangle(legend_img, (10, y_offset-15), (30, y_offset+5), color, -1)
        cv2.rectangle(legend_img, (10, y_offset-15), (30, y_offset+5), (0, 0, 0), 1)
        
        # Draw shape name and piece indices
        text = f"{simple_sig}: {indices}"
        cv2.putText(legend_img, text, (40, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        y_offset += 25
    
    return output, legend_img

