import numpy as np
import numpy.testing as npt
import pytest


def lennard_jones_potential(r, epsilon=1.0, sigma=1.0):
    sr6 = (sigma / r) ** 6
    return 4.0 * epsilon * (sr6**2 - sr6)


def compute_bond_lengths(coords):
    diffs = coords[1:] - coords[:-1]
    return np.linalg.norm(diffs, axis=1)


def boltzmann_weight(energies, kT=1.0):
    shifted = energies - np.min(energies)
    weights = np.exp(-shifted / kT)
    return weights / np.sum(weights)


def test_lj_minimum_position():
    r_values = np.linspace(0.9, 3.0, 1000)
    potentials = lennard_jones_potential(r_values)
    r_min = r_values[np.argmin(potentials)]
    npt.assert_allclose(r_min, 2 ** (1.0 / 6.0), rtol=1e-3)


def test_lj_minimum_energy():
    r_min = 2 ** (1.0 / 6.0)
    v_min = lennard_jones_potential(r_min)
    npt.assert_allclose(v_min, -1.0, atol=1e-12)


def test_lj_large_distance_vanishes():
    v_far = lennard_jones_potential(100.0)
    npt.assert_allclose(v_far, 0.0, atol=1e-8)


def test_bond_lengths_linear_molecule():
    coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]])
    lengths = compute_bond_lengths(coords)
    npt.assert_allclose(lengths, [1.5, 1.5], rtol=1e-12)


def test_boltzmann_weights_sum_to_one():
    energies = np.array([-2.5, -1.0, 0.0, 1.5])
    weights = boltzmann_weight(energies, kT=0.5)
    npt.assert_allclose(np.sum(weights), 1.0, atol=1e-14)


def test_boltzmann_lowest_energy_dominant():
    energies = np.array([0.0, 5.0, 10.0])
    weights = boltzmann_weight(energies, kT=1.0)
    npt.assert_allclose(weights[0], 1.0, atol=1e-4)


def test_bond_lengths_triangle():
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0]])
    lengths = compute_bond_lengths(coords)
    npt.assert_allclose(lengths, [1.0, 1.0], rtol=1e-10)


def test_lj_symmetry_pair():
    r1, r2 = 1.2, 1.8
    v1 = lennard_jones_potential(r1)
    v2 = lennard_jones_potential(r2)
    assert v1 > v2
