import random

import numpy as np
import torch

torch.manual_seed(123)


class Simulator:
    def __init__(self, dim):
        self.weights = torch.randn(dim, dim)

    def run_simulation(self, steps):
        results = []
        state = torch.randn(self.weights.shape[0])

        for i in range(steps):
            noise = np.random.normal(0, 0.1, state.shape)
            state = torch.matmul(self.weights, state) + torch.from_numpy(noise).float()

            if random.random() > 0.5:
                state = state * 0.95

            results.append(state.mean().item())

        return results


sim = Simulator(50)
output = sim.run_simulation(100)
