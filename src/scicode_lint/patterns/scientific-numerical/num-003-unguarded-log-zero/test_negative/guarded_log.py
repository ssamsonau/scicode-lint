import numpy as np


def richter_magnitude(seismic_amplitude, reference_amplitude=1e-6):
    amplitude = np.maximum(seismic_amplitude, 1e-12)
    return np.log10(amplitude / reference_amplitude)


def permeability_from_porosity(porosity):
    valid = porosity > 0
    result = np.zeros_like(porosity)
    result[valid] = np.log(porosity[valid] / (1 - porosity[valid] + 1e-9))
    return result


def acoustic_impedance_log(impedance_series):
    clipped = np.clip(impedance_series, 1.0, None)
    return np.log(clipped)


def radioactive_decay_constant(half_life_years):
    return np.log(2) / half_life_years


amplitudes = np.array([1e-4, 5e-3, 2e-2, 1e-6])
magnitudes = richter_magnitude(amplitudes)

porosity = np.array([0.1, 0.25, 0.4, 0.0, 0.15])
perm = permeability_from_porosity(porosity)
