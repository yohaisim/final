import cv2
import numpy as np
import matplotlib.pyplot as plt
from find_contour import detect_puzzle_pieces  # Changed from detect_pieces
from scipy.spatial import distance
from sklearn.cluster import KMeans
from itertools import combinations


def find_square_corners(contour, num_corners: int = 4):
    contour_points = contour.reshape(-1, 2)

    # centroid
    M = cv2.moments(contour)
    if M["m00"]:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = np.mean(contour_points, axis=0).astype(int)

    # perimeter samples: convex‑hull points preferred (they are already filtered)
    hull = cv2.convexHull(contour)
    perimeter_points = hull.reshape(-1, 2) if len(hull) >= 20 else contour_points

    # cluster those points so we get representative extreme points around the shape
    n_clusters = min(8, len(perimeter_points))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(perimeter_points)

    cluster_corners = []
    for i in range(n_clusters):
        cluster_pts = perimeter_points[kmeans.labels_ == i]
        if len(cluster_pts):
            dists = np.linalg.norm(cluster_pts - np.array([cx, cy]), axis=1)
            cluster_corners.append(cluster_pts[np.argmax(dists)])

    # fallback to extreme points if clusters are not enough
    if len(cluster_corners) < num_corners:
        left = contour_points[np.argmin(contour_points[:, 0])]
        right = contour_points[np.argmax(contour_points[:, 0])]
        top = contour_points[np.argmin(contour_points[:, 1])]
        bottom = contour_points[np.argmax(contour_points[:, 1])]
        return np.array([left, top, right, bottom])

    # evaluate every combination of 4 candidates
    best_score, best_corners = np.inf, None
    sqrt2 = 1.414

    for subset in combinations(cluster_corners, 4):
        corners = np.array(subset)

        # geometric consistency (side / diagonal lengths)
        dists = np.sort(distance.pdist(corners))
        sides, diagonals = dists[:4], dists[4:]
        side_std, diag_std = np.std(sides), np.std(diagonals)
        ratio = np.mean(diagonals) / np.mean(sides) if np.mean(sides) else np.inf

        # prevent corners that are spatially too close
        min_pairwise = np.min([np.linalg.norm(corners[i]-corners[j]) for i in range(4) for j in range(i+1,4)])
        penalty_dist = 1e3 if min_pairwise < 40 else 0

        # angular distribution around centroid – want roughly 90° spacing
        vecs = corners - np.array([cx, cy])
        angles = np.mod(np.arctan2(vecs[:,1], vecs[:,0]), 2*np.pi)
        angles = np.sort(angles)
        diffs = np.diff(np.append(angles, angles[0] + 2*np.pi))
        min_ang = np.min(diffs) * 180/np.pi  # degrees
        penalty_angle = 1e3 if min_ang < 60 else 0

        score = side_std + diag_std + abs(ratio - sqrt2) + penalty_dist + penalty_angle
        if score < best_score:
            best_score, best_corners = score, corners

    if best_corners is None:
        # final fallback
        left = contour_points[np.argmin(contour_points[:, 0])]
        right = contour_points[np.argmax(contour_points[:, 0])]
        top = contour_points[np.argmin(contour_points[:, 1])]
        bottom = contour_points[np.argmax(contour_points[:, 1])]
        best_corners = np.array([left, top, right, bottom])

    return best_corners

def create_corners_visualization(img, contours, all_corners):
    """
    Returns an image with puzzle contours and their 4 detected corners (colored dots and lines).
    """
    out = img.copy()
    cv2.drawContours(out, contours, -1, (0, 255, 0), 2)  # Green contours

    quad_color = (255, 255, 255)  # White lines connecting corners
    pt_colors = [(0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255)]  # Red, Blue, Cyan, Magenta

    for quad in all_corners:
        for i in range(4):
            p1, p2 = tuple(quad[i].astype(int)), tuple(quad[(i + 1) % 4].astype(int))
            cv2.line(out, p1, p2, quad_color, 1)
        for i, p in enumerate(quad):
            cv2.circle(out, tuple(p.astype(int)), 6, pt_colors[i % 4], -1)

    return out



