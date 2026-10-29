"""Explicit closure choices shared by Figures 4, 5, and 6.

Neural models are loaded only on request. The EKR solver retains its
published floor |U_0|<1e-15 and |zeta| cap 12; this is a numerical guard,
not an analytic continuation through density nodes.
"""

import numpy as np

from bot.closed_loop import direct_u2_evolve, pade_evolve
from bot.closures.opt_hunana import opt_coefficients
from bot.closures.pade import pade_coefficients, two_point_coefficients

ORDERS = dict(hp3=3, hunana4=4, opt3=3, opt4=4, beta2=2, nn3h=3, nn4h=4)


def linear_coefficients():
    return dict(
        hp3=pade_coefficients(3),
        hunana4=two_point_coefficients(4, 2),
        opt3=opt_coefficients(3),
        opt4=opt_coefficients(4),
    )


def neural_closures():
    from bot.closures.homog_nn import HomogClosure, load

    return {f"nn{n}h": HomogClosure(load(n)) for n in (3, 4)}


def beam_traces(k, u_b, epsilon, time, initial_states, neural):
    """Return E(t) for seven closures from order-indexed initial states."""
    from bot.closures.homog_nn import bot_evolve

    traces = {
        name: pade_evolve(k, u_b, epsilon, coeff, time, initial_states[len(coeff)], method="rk45")
        for name, coeff in linear_coefficients().items()
    }
    traces["beta2"] = direct_u2_evolve(k, u_b, epsilon, time, initial_states[2])
    for name, closure in neural.items():
        traces[name] = bot_evolve(k, u_b, epsilon, closure, time, initial_states[closure.N])
    return traces


def fit_gamma(time, field, t_min=25.0):
    """Slope of log |E| after t_min, including the published 1e-30 floor."""
    from scipy.stats import linregress

    mask = time >= t_min
    return float(linregress(time[mask], np.log(np.maximum(abs(field[mask]), 1e-30))).slope)
