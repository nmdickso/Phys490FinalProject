# Standard library imports
import itertools

# Dependencies
import numpy as np
import matplotlib.pyplot as plt
import math


def damped_oscillator(k, b, x_0=1, v_0=0, no_points=500, domain=[0, 4 * np.pi]):
    '''
    DESCRIPTION:
        Function to give a timeseries of position and velocity for a
        damped pendulum with given parameters
    REQUIRED ARGUMENTS:
        k:          Spring Constant
        b:          Damping Constant
    OPTIONAL ARGUMENTS:
        x_0:        Initial position of pendulum
        v_0:        Initial velocity of pendulum
        no_points:  Number of time points in series
        domain:     range of time values covered
    '''

    # Initialize array for time and set dt
    t = np.linspace(domain[0], domain[1], no_points)
    dt = t[1] - t[0]

    # Initialize array for position and set initial
    x = np.full((no_points,), np.nan)
    x[0] = x_0

    # Initialize array for velocity and set initial
    v = np.full((no_points,), np.nan)
    v[0] = v_0

    # Solve for all position/velocities
    for i in range(1, no_points):
        v[i] = v[i - 1] + dt * (-k * x[i -1] - b * v[i - 1])
        x[i] = x[i - 1] + dt * v[i]

    # Return time, position and velocity
    return t, x, v


def lorentz_oscillator(k, b, w, E_0=1, x_0=1, v_0=0, no_points=1000, domain=[0, 8 * np.pi]):
    '''
    DESCRIPTION:
        Function to give a timeseries of position and velocity for a
        damped pendulum with given parameters. Modified for the
        Lorentz picture of an electron bound to a molucule in
        an oscillating magnetic field
    REQUIRED ARGUMENTS:
        w_0:        Spring Constant = Resonant Freq of e
        b:          Damping Constant
        E_0:        Amplitude of electric field
        w:          Driving Frequency of electric field
    OPTIONAL ARGUMENTS:
        x_0:        Initial position of pendulum
        v_0:        Initial velocity of pendulum
        no_points:  Number of time points in series
        domain:     range of time values covered
    '''
    # Initialize array for time and set dt
    t = np.linspace(domain[0], domain[1], no_points)
    dt = t[1] - t[0]

    # Initialize array for position and set initial
    x = np.full((no_points,), np.nan)
    x[0] = x_0

    # Initialize array for velocity and set initial
    v = np.full((no_points,), np.nan)
    v[0] = v_0

    # Solve for all position/velocities
    for i in range(1, no_points):
        v[i] = v[i - 1] + dt * (-k * x[i -1] - b * v[i - 1] + E_0 * math.sin(w * t[i]))
        x[i] = x[i - 1] + dt * v[i]

    return t, x, v


def main():
    '''
    DESCRIPTION:
        Generates all damped oscillator timeseries for SciNet
        Requires no arguments to run.
    '''

    all_k = np.linspace(5, 10, 2)
    all_b = np.linspace(0, 1, 2)

    all_kb_pairs = np.array(list(itertools.product(all_k, all_b)))

    for k, b in all_kb_pairs:
        t, x, v = lorentz_oscillator(k, b, 1)
        plt.plot(t, x)
        plt.show()

    # with open("tmp_generated_pingu.csv", 'w') as f:
    #     for k, b in all_kb_pairs:
    #         t, x, v = damped_oscillator(k, b)
    #         # plt.plot(t, x)
    #         # plt.show()
    #         newline = [k, b] + list(x)
    #         newline = ",".join([str(i) for i in newline]) + "\n"
    #         f.write(newline)

    return


if __name__ == "__main__":
    main()


