import numpy as np
from scipy.optimize import minimize_scalar


def bisection_root(func, a, b, tol=1e-8, max_iter=200):
    for _ in range(max_iter):
        mid = (a + b) / 2.0
        if abs(func(mid)) < tol or (b - a) / 2.0 < tol:
            return mid
        if np.sign(func(mid)) == np.sign(func(a)):
            a = mid
        else:
            b = mid
    return (a + b) / 2.0


def gradient_descent(grad_func, x0, lr=0.01, tol=1e-6, max_iter=1000):
    x = np.array(x0, dtype=float)
    for _ in range(max_iter):
        grad = grad_func(x)
        step = lr * grad
        x = x - step
        if np.linalg.norm(step) < tol:
            break
    return x


def running_mean(values):
    total = 0.0
    results = []
    for i, v in enumerate(values):
        total += v
        results.append(total / (i + 1))
    return results


data = np.random.randn(50)
means = running_mean(data)

result = bisection_root(lambda x: x**3 - x - 2, 1.0, 2.0)
