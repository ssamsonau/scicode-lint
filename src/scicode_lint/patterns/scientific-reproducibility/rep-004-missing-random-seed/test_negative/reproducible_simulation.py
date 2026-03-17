"""Reproducible simulation using explicit Generator injection."""

import torch


class DeterministicSimulator:
    """Simulator that accepts a Generator for reproducibility."""

    def __init__(self, n_particles: int, generator: torch.Generator):
        self.n_particles = n_particles
        self.gen = generator
        self.positions = torch.randn(n_particles, 3, generator=self.gen)

    def step(self) -> torch.Tensor:
        """Single simulation step with deterministic noise."""
        noise = torch.randn(self.n_particles, 3, generator=self.gen) * 0.01
        self.positions += noise
        return self.positions.clone()


def run_simulation(seed: int = 42, n_steps: int = 100) -> list[torch.Tensor]:
    """Run simulation with explicit seed control."""
    gen = torch.Generator().manual_seed(seed)
    sim = DeterministicSimulator(50, gen)
    return [sim.step() for _ in range(n_steps)]
