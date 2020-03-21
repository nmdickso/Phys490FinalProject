# Standard library imports
import itertools
import argparse
import math

# Dependencies
import numpy as np
import matplotlib.pyplot as plt

# Script Defaults
DEFAULT_DOMAIN = [0, 4]
DEFAULT_NUMPOINTS = 500
DEFAULT_x0 = 1
DEFAULT_v0 = 0


def damped_oscillator(
    k,
    b,
    domain=DEFAULT_DOMAIN,
    num_points=DEFAULT_NUMPOINTS,
    x_0=DEFAULT_x0,
    v_0=DEFAULT_v0
):
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
        num_points: Number of time points in series
        domain:     range of time values covered
    '''

    # Initialize array for time and set dt
    t = np.linspace(domain[0], domain[1], num_points)
    dt = t[1] - t[0]

    # Initialize array for position and set initial
    x = np.full((num_points,), np.nan)
    x[0] = x_0

    # Initialize array for velocity and set initial
    v = np.full((num_points,), np.nan)
    v[0] = v_0

    # Solve for all position/velocities
    for i in range(1, num_points):
        v[i] = v[i - 1] + dt * (-k * x[i -1] - b * v[i - 1])
        x[i] = x[i - 1] + dt * v[i]

    # Return time, position and velocity
    return t, x, v


def main(
    output_file,
    k_tup,
    b_tup,
    domain=DEFAULT_DOMAIN,
    num_points=DEFAULT_NUMPOINTS,
    x_0=DEFAULT_x0,
    v_0=DEFAULT_v0
):
    '''
    DESCRIPTION:
        Generates all damped oscillator timeseries for SciNet
        Requires no arguments to run.
    '''

    t = np.linspace(domain[0], domain[1], num_points)
    all_k = np.linspace(*k_tup)
    all_b = np.linspace(*b_tup)

    all_kb_pairs = np.array(list(itertools.product(all_k, all_b)))

    # OUTPUT FILE FORMAT:
    #   line 1: time values
    #   all rest: k, b, pos1, pos2, .... etc
    with open(output_file, 'w') as f:
        str_t = [f"{i:.6f}" for i in t]
        newline = " ".join(str_t) + "\n"
        f.write(newline)        
        for k, b in all_kb_pairs:
            _, x, _ = damped_oscillator(k, b, domain=domain, num_points=num_points, x_0=x_0, v_0=v_0)
            newline = [f"{k:.6f}", f"{b:.6f}"] + [f"{i:.6f}" for i in x]
            newline = " ".join(newline) + "\n"
            f.write(newline)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='To generate damped oscillator values for SciNet examples')
    parser.add_argument(
        'output_file',
        type=str,
        help="Path and name of file to save output data",
    )
    parser.add_argument(
        '--k_range',
        nargs=3,
        type=float,
        help='range and number of spring constants, e.g. (min, max, total number)',
        required=False,
        default=[5, 10, 25]
    )
    parser.add_argument(
        '--b_range',
        nargs=3,
        type=float,
        help='range and number of damping constants, e.g. (min, max, total number)',
        required=False,
        default=[0, 1, 25]
    )
    parser.add_argument(
        '-d',
        '--domain',
        nargs=2,
        type=float,
        help="time domain to solve position time series over",
        required=False,
        default=DEFAULT_DOMAIN
    )
    parser.add_argument(
        '-np',
        '--num_points',
        type=int,
        help="Number of points in the output time series",
        required=False,
        default=DEFAULT_NUMPOINTS
    )
    parser.add_argument(
        '-x0',
        '--initial_position',
        type=float,
        help="Initial position of damped oscillator",
        required=False,
        default=DEFAULT_x0
    )
    parser.add_argument(
        '-v0',
        '--initial_velocity',
        type=float,
        help="Initial velocity of damped oscillator",
        required=False,
        default=DEFAULT_v0
    )
    args = parser.parse_args()

    output_file = args.output_file
    k_tup = args.k_range
    k_tup = [k_tup[0], k_tup[1], int(k_tup[2])]
    b_tup = args.b_range
    b_tup = [b_tup[0], b_tup[1], int(b_tup[2])]
    domain = args.domain
    num_points = args.num_points
    x_0 = args.initial_position
    v_0 = args.initial_velocity

    main(output_file, k_tup, b_tup, domain=domain, num_points=num_points, x_0=x_0, v_0=v_0)


