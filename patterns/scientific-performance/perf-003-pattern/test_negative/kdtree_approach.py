import numpy as np
from scipy.spatial import KDTree


def find_nearby_points(points, query_points, radius):
    tree = KDTree(points)
    neighbors = []
    for query in query_points:
        indices = tree.query_ball_point(query, radius)
        neighbors.append(indices)
    return neighbors


all_points = np.random.randn(50000, 3)
queries = np.random.randn(100, 3)
nearby = find_nearby_points(all_points, queries, radius=0.5)
