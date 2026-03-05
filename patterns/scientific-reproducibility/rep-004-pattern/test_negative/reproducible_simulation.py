import random

import torch

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)


class MonteCarloSimulator:
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.positions = torch.randn(n_agents, 2)
        self.velocities = torch.randn(n_agents, 2) * 0.1

    def step(self):
        random_force = torch.randn_like(self.positions) * 0.05

        for i in range(self.n_agents):
            if random.random() > 0.7:
                self.velocities[i] *= 0.9

        self.velocities += random_force
        self.positions += self.velocities

    def run(self, steps):
        trajectory = []
        for _ in range(steps):
            self.step()
            trajectory.append(self.positions.clone())
        return torch.stack(trajectory)


sim = MonteCarloSimulator(100)
results = sim.run(500)
