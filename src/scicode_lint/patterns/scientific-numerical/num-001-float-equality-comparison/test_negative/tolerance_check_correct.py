import numpy as np


def softmax(logits):
    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum()


def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v.copy().astype(float)
        for b in basis:
            w -= np.dot(w, b) * b
        norm = np.linalg.norm(w)
        if norm > 1e-10:
            basis.append(w / norm)
    return np.array(basis)


def power_iteration(matrix, num_iterations=100, tol=1e-9):
    n = matrix.shape[0]
    b = np.random.rand(n)
    eigenvalue = 0.0
    for _ in range(num_iterations):
        b_new = matrix @ b
        new_eigenvalue = np.linalg.norm(b_new)
        b = b_new / new_eigenvalue
        if abs(new_eigenvalue - eigenvalue) < tol:
            break
        eigenvalue = new_eigenvalue
    return eigenvalue, b


A = np.array([[4.0, 1.0], [2.0, 3.0]])
eigenvalue, eigenvec = power_iteration(A)

vecs = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
orthonormal = gram_schmidt(vecs)
