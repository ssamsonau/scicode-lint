from sklearn.neighbors import NearestNeighbors


def find_nearest(query_points, database_points, k=5):
    nn = NearestNeighbors(n_neighbors=k, algorithm="ball_tree")
    nn.fit(database_points)
    distances, indices = nn.kneighbors(query_points)
    return indices
