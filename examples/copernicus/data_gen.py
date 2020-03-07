import numpy as np

# Orbital periods [days]
T_e = 365.26
T_m = 686.97959

# Orbit semi-major axis [AU]
a_e = 1.
a_m = 1.52366231


def generate_orbits(N, M, del_t):
    '''
    N = N_dataset, amount of training datasets
    M = N_timeseries, amount of time evolution iterations
    del_t = amount of time (in days) for each time evolution iteration
    '''

    # Generate initial mean anomalies
    φ_e0 = (2.4 * np.pi) * np.random.rand(N) - (0.2 * np.pi)
    φ_m0 = (2. * np.pi) * np.random.rand(N)

    φ_e0 = φ_e0[:, np.newaxis].repeat(M, axis=1)
    φ_m0 = φ_m0[:, np.newaxis].repeat(M, axis=1)

    # Compute Mean anomalies (φ)
    time_series = np.arange(M)[np.newaxis, :].repeat(N, axis=0)

    φ_e = φ_e0 + (2 * np.pi / T_e) * (time_series * del_t)
    φ_m = φ_m0 + (2 * np.pi / T_m) * (time_series * del_t)

    # Compute Earth angles

    d = (a_e**2 + a_m**2 - 2 * a_e * a_m * np.cos(φ_m - φ_e))  # cosine law

    sin_θ = ((a_e * np.sin(φ_e)) + (a_m * np.sin(φ_m))) / d    # earth-mars Δ
    cos_θ = ((a_e * np.cos(φ_e)) + (a_m * np.cos(φ_m))) / d

    θ_m = np.angle(cos_θ + 1j * sin_θ)
    θ_e = φ_e

    return φ_e, φ_m, θ_e, θ_m

