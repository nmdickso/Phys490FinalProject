import numpy as np

# Orbital periods [days]
T_e = 365.26
T_m = 686.97959

# Orbit semi-major axis [AU]
a_e = 1.
a_m = 1.52366231


def anim_orbit(φ_e, φ_m, θ_e, θ_m):
    '''show an animated orbit example, of both viewpoints
    only uses the first row of the datasets, for ease
    '''
    import matplotlib.pyplot as plt
    import matplotlib.animation as anim

    def _update_plot(n):

        lines[0].set_data(φ_e[:n], R_e[:n])
        lines[1].set_data(φ_m[:n], R_m[:n])

        lines[2].set_data(θ_e[:n], R_e[:n])
        lines[3].set_data(θ_m[:n], R_m[:n])

        lines[4].set_data(x[:n], φ_e[:n])
        lines[5].set_data(x[:n], φ_m[:n])

        lines[6].set_data(x[:n], θ_e[:n])
        lines[7].set_data(x[:n], θ_m[:n])

        return lines

    φ_e, φ_m, θ_e, θ_m = φ_e[0, :], φ_m[0, :], θ_e[0, :], θ_m[0, :]

    R_e, R_m = np.ones_like(φ_e) * a_e, np.ones_like(φ_m) * a_m
    x = range(φ_e.size)

    fig = plt.figure()

    ax1, ax2 = fig.add_subplot(221, polar=1), fig.add_subplot(222, polar=1)
    ax3, ax4 = fig.add_subplot(223), fig.add_subplot(224)

    ax1.set_title('Sun Angle (φ)')
    ax2.set_title('Earth Angle (θ)')

    ax1.plot(0, 0, 'ko')
    [line1_e] = ax1.plot(φ_e, R_e, '-go', markevery=[-1])
    [line1_m] = ax1.plot(φ_m, R_m, '-ro', markevery=[-1])

    ax2.plot(0, 0, 'go')
    [line2_e] = ax2.plot(θ_e, R_e, '-ko', markevery=[-1])
    [line2_m] = ax2.plot(θ_m, R_m, '-ro', markevery=[-1])

    [line3_e] = ax3.plot(x, φ_e, '-go')
    [line3_m] = ax3.plot(x, φ_m, '-ro')

    [line4_e] = ax4.plot(x, θ_e, '-ko')
    [line4_m] = ax4.plot(x, θ_m, '-ro')

    lines = [line1_e, line1_m, line2_e, line2_m,
             line3_e, line3_m, line4_e, line4_m]

    fig.legend(lines, ['Earth', 'Mars', 'Sun'])

    # TODO cool little angle (shading) between planets to see plot relations
    # d = (a_e**2 + a_m**2 - 2 * a_e * a_m * np.cos(φ_m - φ_e))

    anim.FuncAnimation(fig, _update_plot, φ_e.size, interval=10, blit=True)

    plt.show()


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
