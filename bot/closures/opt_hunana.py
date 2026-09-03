"""Optimized Hunana-style linear closure: coefficients fit to the resonant
response function on the SAME complex rectangle used to train the NN closures.

Motivation
----------
The Hammett-Perkins / Hunana family closes the beam moment hierarchy with a
constant linear map

    U_N = sum_{i=0}^{N-1} a_i U_i,

whose only content is the rational response function it induces,

    R_N(xi; a) = U_0/Phi,   targeting the exact kinetic  Z'(xi).

HP (N=3) fixes `a` by matching R_N to Z' at exactly two points (the xi -> inf
asymptotics, plus Im Z'(0) for the Landau damping rate); `pade_coefficients(N)`
for N != 3 Taylor-matches at the single point xi = 0; `aaa_coefficients_n4`
fits along the REAL axis only.  All three are therefore optimal somewhere the
closure is not actually used: the modes that matter sit at complex xi_b with
Im(xi_b) != 0.

The NN closures (bot.closures.u2, bot.closures.n4_nn) are trained on xi_b drawn
uniformly from the complex rectangle

    Re(xi_b) in (-3, 3),   Im(xi_b) in (-1, 1.5),

so any "linear closure vs NN closure" comparison that uses HP or Padé
coefficients confounds two things: the architecture (linear-in-the-moments vs
nonlinear) and the fitting domain (2 points / real axis vs the rectangle).
This module removes the second by fitting `a` over exactly that rectangle --
giving the linear closure the same information the NN got -- so what is left is
the architecture comparison.

Two equivalent handles on the response
--------------------------------------
With Phi = 1 the hierarchy  xi U_n - U_{n+1} = n M_{n-1}  gives
U_n = xi^n U_0 - q_n(xi), where

    q_0 = 0,    q_{n+1}(xi) = xi q_n(xi) + n M_{n-1}

(so q_1 = 0, q_2 = 1, q_3 = xi, q_4 = xi^2 + 3/2, ...).  Substituting into the
closure row  xi U_{N-1} - sum_i a_i U_i = (N-1) M_{N-2}  yields the closed
rational response

    (R)   R_N(xi; a) = [ q_N(xi) - sum_i a_i q_i(xi) ] / [ xi^N - sum_i a_i xi^i ]

-- for N=3 this is (xi - a_2)/(xi^3 - a_2 xi^2 - a_1 xi - a_0), matching the
bot.closures.pade docstring.  The same information sits in the closure ratio
evaluated on the kinetic manifold, U_n/U_0 = alpha_n(xi) = xi^n - q_n(xi)/Z'(xi):

    (A)   alpha_N_hat(xi; a) = sum_i a_i alpha_i(xi)     vs     alpha_N(xi)

(R) and (A) both say "reproduce Z' on the rectangle" -- alpha_N is an exact,
invertible transform of Z' at fixed xi -- but they behave very differently as
fitting objectives:

  * (A) is LINEAR in `a` (ordinary least squares, global optimum, no restarts)
    and pole-free: alpha_N is analytic wherever Z' has no zeros, and Z' has
    none on this rectangle (its nearest zeros are at xi ~ +-2.0 - 1.35i).
    It is also exactly the quantity the NN regresses, so fitting it puts the
    linear closure and the NN on the same loss over the same samples.
  * (R) is nonlinear in `a` and every member of the family has N poles of its
    own, which for N >= 3 land INSIDE the rectangle (Im ~ -0.3 to -1.2).  Near
    them |R_N - Z'| diverges while Z' stays finite, so a plain least-squares
    objective is dominated by a handful of pole-adjacent samples and the fit
    degrades into pole-repulsion.  A robust loss controls this but the result
    is still worse than (A) at every N tested.

Default is therefore `target='alpha'`; `target='response'` implements (R)
directly (robust loss) for comparison.  The irreducible pole scars are a
property of the rational family, not of the fit: they are what the response
error maps in bot/figs/fig_opt_hunana_response.py show.

Parity constraint
-----------------
Z'(-conj(xi)) = conj(Z'(xi)), and the manifold ratios inherit
alpha_n(-conj(xi)) = (-1)^n conj(alpha_n(xi)).  A closure respecting this
symmetry must satisfy a_i = (-1)^(N-i) conj(a_i), i.e.

    a_i real           if (N - i) is even,
    a_i pure imaginary  if (N - i) is odd.

HP's (i/sqrt(pi), 3/2, -2i/sqrt(pi)) and `aaa_coefficients_n4` both obey it.
Imposing it exactly halves the parameter count (N reals instead of 2N) and
guarantees the fitted response inherits the kinetic symmetry, so e.g. growth
rates come out identical for +k and -k.  It costs nothing in accuracy: the
unconstrained fits (parity=False) land on the same coefficients to ~1e-3.

Stability
---------
The denominator roots of (R) are the closure's own (free-streaming) poles; a
usable closure needs them all in the LOWER half plane, else the closed fluid
system has a spurious growing mode.  Every fit here is checked, and a
penalized nonlinear refit is triggered if the least-squares optimum violates
it (it does not, for the settings shipped in `_CACHED`).

Scope -- these coefficients are tuned for BoT, not for slab ITG
---------------------------------------------------------------
The fit targets the universal gradient-free response Z'(xi).  That is the
right object for the bump-on-tail system (bot.fluid.fluid_system), where
`opt_coefficients(3)` cuts the growth-rate error against the kinetic reference
by roughly 5x relative to HP (median over unstable k: 4.9% -> 1.0% at
u_b=5, eps=0.05; 13.7% -> 4.5% at u_b=4, eps=0.02).

It is NOT the right object for slab ITG.  There the dispersion relation is set
by R_kinetic(zeta; zeta_*, eta) (bot.slab_itg), whose eta-drive terms weight
Z-moments the universal Z' fit never sees, and closure quality in Z' turns out
to be nearly uncorrelated with ITG growth-rate accuracy: at zeta_*=tau=1,
eta=8, HP N=3 has the WORST Z' error at the mode frequency (9.0%) yet the best
growth rate (-9.3%), while Padé N=4 has 0.29% Z' error and -22.3% growth-rate
error.  Convergence in N is non-monotonic there (N=3: -9.3%, N=4: -22.3%,
N=5: -7.4%, N=6: -2.4%, N=8: -0.2%).  For an ITG-specific fixed linear
closure, fit against R_kinetic instead -- see bot.closures.aaa_pad.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.special import wofz

from bot.closures.pade import (maxwellian_moment, pade_coefficients,
                               aaa_coefficients_n4)


SQRT_PI = float(np.sqrt(np.pi))

# The NN training rectangle (bot.closures.n4_nn.make_dataset_n4,
# bot.closures.u2.make_dataset_u2) -- the domain this module fits over.
NN_RE_RANGE = (-3.0, 3.0)
NN_IM_RANGE = (-1.0, 1.5)


def Z(xi):
    return 1j * SQRT_PI * wofz(xi)


def Zprime(xi):
    return -2.0 * (1.0 + xi * Z(xi))


# ---------------------------------------------------------------------------
# Rational response and closure ratio of a linear closure
# ---------------------------------------------------------------------------

def q_polynomials(N: int) -> list[np.ndarray]:
    """q_0..q_N as coefficient arrays in INCREASING powers of xi.

    q_0 = 0,  q_{n+1} = xi q_n + n M_{n-1}.
    """
    qs = [np.zeros(1)]
    for n in range(N):
        prev = qs[-1]
        nxt = np.zeros(len(prev) + 1)
        nxt[1:] = prev                              # xi * q_n
        nxt[0] += n * maxwellian_moment(n - 1)
        qs.append(nxt)
    return qs


def _polyval_inc(coeffs: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Evaluate a polynomial given in increasing-power order."""
    return np.polyval(coeffs[::-1].astype(complex), xi)


def response(xi, a) -> np.ndarray:
    """R_N(xi; a) = U_0/Phi for the closure U_N = sum a_i U_i.

    Vectorized closed form; agrees with bot.closures.pade.fluid_U0 (which
    solves the N x N moment system pointwise) to round-off.
    """
    a = np.asarray(a, dtype=complex)
    N = len(a)
    xi = np.asarray(xi, dtype=complex)
    qs = q_polynomials(N)

    num = _polyval_inc(qs[N], xi)
    den = xi**N
    for i in range(N):
        if a[i] != 0:
            num = num - a[i] * _polyval_inc(qs[i], xi)
        den = den - a[i] * xi**i
    return num / den


def closure_poles(a) -> np.ndarray:
    """Roots of xi^N - sum_i a_i xi^i: the closure's own poles.

    Stability/causality of the closed fluid system requires Im < 0 for all of
    them (time dependence exp(-i xi t)).
    """
    a = np.asarray(a, dtype=complex)
    N = len(a)
    coeffs = np.zeros(N + 1, dtype=complex)
    coeffs[0] = 1.0                                 # decreasing-power order
    coeffs[1:] = -a[::-1]
    return np.roots(coeffs)


def alpha_basis(xi, N: int) -> tuple[np.ndarray, np.ndarray]:
    """(B, y): B[:, i] = alpha_i(xi) for i < N, y = alpha_N(xi).

    alpha_n(xi) = xi^n - q_n(xi)/Z'(xi) is the exact U_n/U_0 on the kinetic
    eigenmode manifold: alpha_0 = 1, alpha_1 = xi, alpha_2 = beta (u2.beta),
    alpha_3 = alpha (inference.alpha), alpha_4 = alpha4 (inference.alpha4).
    """
    xi = np.asarray(xi, dtype=complex)
    qs = q_polynomials(N)
    Zp = Zprime(xi)
    B = np.stack([xi**i - _polyval_inc(qs[i], xi) / Zp for i in range(N)],
                 axis=-1)
    y = xi**N - _polyval_inc(qs[N], xi) / Zp
    return B, y


def implied_alpha(xi, a) -> np.ndarray:
    """Closure value U_N/U_0 = sum_i a_i alpha_i(xi) predicted by `a`.

    The head-to-head metric against the NN, whose training target IS alpha_N;
    `response` is the Hunana-style metric.
    """
    a = np.asarray(a, dtype=complex)
    B, _ = alpha_basis(xi, len(a))
    return B @ a


def exact_alpha_n(xi, N: int) -> np.ndarray:
    """Exact U_N/U_0 on the manifold: xi^N - q_N(xi)/Z'(xi)."""
    return alpha_basis(xi, N)[1]


# ---------------------------------------------------------------------------
# Parity parameterization
# ---------------------------------------------------------------------------

def parity_mask(N: int) -> np.ndarray:
    """Unit multipliers m_i with a_i = t_i * m_i, t_i real: 1 or 1j."""
    return np.array([1.0 + 0j if (N - i) % 2 == 0 else 1j for i in range(N)])


def pack(a, parity: bool = True) -> np.ndarray:
    """Complex coefficients -> real parameter vector."""
    a = np.asarray(a, dtype=complex)
    if not parity:
        return np.concatenate([a.real, a.imag])
    return (a / parity_mask(len(a))).real


def unpack(t: np.ndarray, N: int, parity: bool = True) -> np.ndarray:
    """Real parameter vector -> complex coefficients."""
    t = np.asarray(t, dtype=float)
    if not parity:
        return t[:N] + 1j * t[N:]
    return t * parity_mask(N)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_rectangle(n: int = 20000,
                     re_range: tuple[float, float] = NN_RE_RANGE,
                     im_range: tuple[float, float] = NN_IM_RANGE,
                     seed: int = 0) -> np.ndarray:
    """Uniform samples of xi over the NN training rectangle.

    Same distribution as make_dataset_n4 / make_dataset_u2 (uniform in the
    rectangle, independent Re and Im), so the linear fit sees the same measure
    over xi that the NN's MSE loss does.
    """
    rng = np.random.default_rng(seed)
    re = rng.uniform(*re_range, size=n)
    im = rng.uniform(*im_range, size=n)
    return re + 1j * im


def _weights(target_vals: np.ndarray, mode: str, delta: float) -> np.ndarray:
    if mode == "absolute":
        return np.ones(target_vals.shape, dtype=float)
    if mode == "relative":
        return 1.0 / (np.abs(target_vals) + delta)
    raise ValueError(f"unknown weight mode {mode!r}")


# ---------------------------------------------------------------------------
# Fit result
# ---------------------------------------------------------------------------

@dataclass
class OptFit:
    """Result of a coefficient fit."""
    N: int
    a: np.ndarray
    target: str
    weight: str
    parity: bool
    n_samples: int
    re_range: tuple[float, float]
    im_range: tuple[float, float]
    cost: float
    poles: np.ndarray
    stable: bool
    metrics: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        coef = ", ".join(f"{c.real:+.6f}{c.imag:+.6f}j" for c in self.a)
        return (f"OptFit(N={self.N}, target={self.target!r}, "
                f"weight={self.weight!r}, stable={self.stable}, "
                f"cost={self.cost:.4e},\n       a=[{coef}])")


# ---------------------------------------------------------------------------
# (A) closure-ratio target: ordinary least squares, global optimum
# ---------------------------------------------------------------------------

def _fit_alpha_ls(N: int, xi: np.ndarray, weight: str, delta: float,
                  parity: bool) -> tuple[np.ndarray, float]:
    """Least-squares fit of sum_i a_i alpha_i(xi) to alpha_N(xi).

    Linear in `a`, so this is a single lstsq -- no restarts, no local minima.
    With parity=True the design matrix is real-ified as a = t * m (t real,
    m the parity mask), which is exactly the symmetry-constrained problem.
    """
    B, y = alpha_basis(xi, N)
    w = _weights(y, weight, delta)
    Bw, yw = B * w[:, None], y * w

    if parity:
        m = parity_mask(N)
        A = np.concatenate([(Bw * m[None, :]).real, (Bw * m[None, :]).imag])
        b = np.concatenate([yw.real, yw.imag])
        t, *_ = np.linalg.lstsq(A, b, rcond=None)
        a = t * m
    else:
        A = np.block([[Bw.real, -Bw.imag], [Bw.imag, Bw.real]])
        b = np.concatenate([yw.real, yw.imag])
        v, *_ = np.linalg.lstsq(A, b, rcond=None)
        a = v[:N] + 1j * v[N:]

    resid = (B @ a - y) * w
    return a, 0.5 * float(np.sum(np.abs(resid) ** 2))


def _alpha_residuals(t: np.ndarray, N: int, B: np.ndarray, y: np.ndarray,
                     w: np.ndarray, parity: bool, pole_penalty: float,
                     pole_margin: float) -> np.ndarray:
    a = unpack(t, N, parity)
    d = (B @ a - y) * w
    res = np.concatenate([d.real, d.imag])
    if pole_penalty > 0.0:
        viol = np.maximum(closure_poles(a).imag + pole_margin, 0.0)
        res = np.concatenate([res, pole_penalty * viol])
    return res


# ---------------------------------------------------------------------------
# (R) response target: robust nonlinear least squares
# ---------------------------------------------------------------------------

def _response_residuals(t: np.ndarray, N: int, xi: np.ndarray,
                        target_vals: np.ndarray, w: np.ndarray, parity: bool,
                        pole_penalty: float, pole_margin: float) -> np.ndarray:
    a = unpack(t, N, parity)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        r = response(xi, a)
    d = (r - target_vals) * w
    d = np.where(np.isfinite(d), d, 1e3)            # a pole hit a sample point
    res = np.concatenate([d.real, d.imag])
    if pole_penalty > 0.0:
        viol = np.maximum(closure_poles(a).imag + pole_margin, 0.0)
        res = np.concatenate([res, pole_penalty * viol])
    return res


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def fit_coefficients(N: int,
                     target: str = "alpha",
                     weight: str = "relative",
                     delta: float = 0.0,
                     re_range: tuple[float, float] = NN_RE_RANGE,
                     im_range: tuple[float, float] = NN_IM_RANGE,
                     n_samples: int = 20000,
                     parity: bool = True,
                     pole_penalty: float = 5.0,
                     pole_margin: float = 0.02,
                     loss: str = "soft_l1",
                     f_scale: float = 1.0,
                     n_restarts: int = 12,
                     seed: int = 0,
                     verbose: bool = False) -> OptFit:
    """Fit the closure coefficients to the kinetic response on the rectangle.

    target='alpha' (default)
        Minimize sum |sum_i a_i alpha_i(xi) - alpha_N(xi)|^2 w^2 over xi
        sampled uniformly from the rectangle.  Linear in `a` -> exact global
        optimum from one lstsq.  This is the NN's own loss and target.
    target='response'
        Minimize sum rho(|R_N(xi; a) - Z'(xi)|^2 w^2) with a robust loss rho
        (`loss`/`f_scale` are passed to scipy.optimize.least_squares), from a
        multi-start over HP/Padé/AAA plus random perturbations.  The robust
        loss is needed because the closure's own poles sit inside the sampling
        rectangle; see the module docstring.

    weight='relative' uses w = 1/(|target| + delta), 'absolute' uses w = 1
    (the NN's unweighted MSE).  delta=0 is safe on this rectangle: neither
    alpha_N nor Z' vanishes on it.

    If the fit lands with a pole in the upper half plane, a penalized
    nonlinear refit is run from that point (`pole_penalty`, `pole_margin`).
    """
    xi = sample_rectangle(n_samples, re_range, im_range, seed=seed)

    if target == "alpha":
        B, y = alpha_basis(xi, N)
        w = _weights(y, weight, delta)
        a, cost = _fit_alpha_ls(N, xi, weight, delta, parity)
        poles = closure_poles(a)
        if not np.all(poles.imag < 0.0):            # rare; repair
            sol = least_squares(
                _alpha_residuals, pack(a, parity),
                args=(N, B, y, w, parity, pole_penalty, pole_margin),
                method="trf", xtol=1e-14, ftol=1e-14, gtol=1e-14)
            a = unpack(sol.x, N, parity)
            cost = float(sol.cost)
            poles = closure_poles(a)

    elif target == "response":
        tgt = Zprime(xi)
        w = _weights(tgt, weight, delta)

        starts: list[np.ndarray] = []
        try:
            starts.append(pade_coefficients(N))
        except Exception:
            pass
        if N == 4:
            starts.append(aaa_coefficients_n4())
        starts.append(_fit_alpha_ls(N, xi, weight, delta, parity)[0])

        rng = np.random.default_rng(seed + 1)
        base = starts[0]
        while len(starts) < n_restarts:
            scale = np.maximum(np.abs(base), 1.0)
            starts.append(base + rng.normal(0.0, 1.0, size=N) * scale
                          * parity_mask(N))

        best_a, best_cost, best_ok = None, np.inf, False
        for a0 in starts:
            try:
                sol = least_squares(
                    _response_residuals, pack(np.asarray(a0, complex), parity),
                    args=(N, xi, tgt, w, parity, pole_penalty, pole_margin),
                    loss=loss, f_scale=f_scale, method="trf",
                    xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=4000)
            except Exception:
                continue
            if not np.all(np.isfinite(sol.x)):
                continue
            a_i = unpack(sol.x, N, parity)
            ok = bool(np.all(closure_poles(a_i).imag < 0.0))
            if verbose:
                print(f"    start -> cost={sol.cost:.4e} stable={ok}")
            if (ok, -sol.cost) > (best_ok, -best_cost):
                best_a, best_cost, best_ok = a_i, float(sol.cost), ok
        if best_a is None:
            raise RuntimeError(f"all restarts failed for N={N}")
        a, cost, poles = best_a, best_cost, closure_poles(best_a)

    else:
        raise ValueError(f"unknown target {target!r}")

    fit = OptFit(N=N, a=a, target=target, weight=weight, parity=parity,
                 n_samples=n_samples, re_range=re_range, im_range=im_range,
                 cost=cost, poles=poles, stable=bool(np.all(poles.imag < 0.0)))
    fit.metrics = {**response_metrics(a, re_range, im_range, seed=seed + 7),
                   **alpha_metrics(a, re_range, im_range, seed=seed + 7)}
    return fit


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def response_metrics(a, re_range: tuple[float, float] = NN_RE_RANGE,
                     im_range: tuple[float, float] = NN_IM_RANGE,
                     n_samples: int = 20000, seed: int = 123) -> dict:
    """Relative response error |R_N - Z'|/|Z'| on the rectangle and real axis.

    The p90/max tails are dominated by the closure's own poles inside the
    rectangle and are reported for that reason, not as fit quality.
    """
    xi = sample_rectangle(n_samples, re_range, im_range, seed=seed)
    tgt = Zprime(xi)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        rel = np.abs(response(xi, a) - tgt) / np.abs(tgt)
    rel = rel[np.isfinite(rel)]

    xr = np.linspace(re_range[0], re_range[1], 2001).astype(complex)
    tr = Zprime(xr)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        rel_r = np.abs(response(xr, a) - tr) / np.abs(tr)
    rel_r = rel_r[np.isfinite(rel_r)]

    return {"R_median": float(np.median(rel)),
            "R_p90": float(np.percentile(rel, 90)),
            "R_axis_median": float(np.median(rel_r)),
            "R_axis_max": float(np.max(rel_r))}


def alpha_metrics(a, re_range: tuple[float, float] = NN_RE_RANGE,
                  im_range: tuple[float, float] = NN_IM_RANGE,
                  n_samples: int = 20000, seed: int = 123) -> dict:
    """Closure-value error |alpha_N_hat - alpha_N| on the rectangle.

    The NN's own training metric, so this is the like-for-like comparison.
    """
    a = np.asarray(a, dtype=complex)
    xi = sample_rectangle(n_samples, re_range, im_range, seed=seed)
    B, exact = alpha_basis(xi, len(a))
    err = np.abs(B @ a - exact)
    rel = err / np.abs(exact)
    return {"A_rms": float(np.sqrt(np.mean(err ** 2))),
            "A_median": float(np.median(err)),
            "A_rel_median": float(np.median(rel)),
            "A_rel_p90": float(np.percentile(rel, 90))}


def metrics_table(entries: dict[str, np.ndarray], **kw) -> str:
    """Formatted response/closure error table for a dict {label: a}."""
    head = (f"{'closure':<22} {'R med':>8} {'R p90':>8} {'R axis':>8} "
            f"{'alpha med':>10} {'alpha p90':>10} {'stable':>7}")
    lines = [head, "-" * len(head)]
    for label, a in entries.items():
        m = {**response_metrics(a, **kw), **alpha_metrics(a, **kw)}
        stable = bool(np.all(closure_poles(a).imag < 0.0))
        lines.append(f"{label:<22} {m['R_median']:>8.2%} {m['R_p90']:>8.2%} "
                     f"{m['R_axis_median']:>8.2%} {m['A_rel_median']:>10.2%} "
                     f"{m['A_rel_p90']:>10.2%} {str(stable):>7}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cached fits (regenerate with `python -m bot.closures.opt_hunana`)
# ---------------------------------------------------------------------------

# target='alpha', weight='relative' (delta=0), parity-constrained, 20000
# uniform samples on Re in (-3,3) x Im in (-1,1.5), seed=0.
_CACHED: dict[int, np.ndarray] = {
    3: np.array([+0.0000000000 + 0.8796497230j,
                 +2.0218845575 + 0.0000000000j,
                 -0.0000000000 - 1.7447864064j]),
    4: np.array([-1.9622890281 + 0.0000000000j,
                 +0.0000000000 + 4.9479177508j,
                 +5.2957359968 + 0.0000000000j,
                 -0.0000000000 - 2.9417426098j]),
    5: np.array([-0.0000000000 - 5.6160786331j,
                 -15.3355419368 + 0.0000000000j,
                 +0.0000000000 + 18.3035006665j,
                 +12.2058910680 + 0.0000000000j,
                 -0.0000000000 - 4.7123866497j]),
    6: np.array([+16.4670385148 + 0.0000000000j,
                 -0.0000000000 - 48.1820393480j,
                 -62.9508365196 + 0.0000000000j,
                 +0.0000000000 + 47.6575511348j,
                 +22.4856508139 + 0.0000000000j,
                 -0.0000000000 - 6.4854786151j]),
}


def opt_coefficients(N: int) -> np.ndarray:
    """Cached optimized coefficients for the N-moment closure.

    Drop-in replacement for pade_coefficients(N) / aaa_coefficients_n4() --
    same `a` convention (U_N = sum_i a_i U_i), so it works unchanged in
    bot.fluid.fluid_system, bot.closed_loop.pade_evolve and
    bot.slab_itg.fluid_pade_system.
    """
    if N not in _CACHED:
        raise KeyError(f"no cached optimized coefficients for N={N}; call "
                       f"fit_coefficients({N}) or rerun the __main__ driver")
    return _CACHED[N].copy()


def _print_report(fit: OptFit) -> None:
    print(f"\n=== N={fit.N}  target={fit.target}  weight={fit.weight} ===")
    print(f"  a = {np.array2string(fit.a, precision=10, separator=', ')}")
    print(f"  cost = {fit.cost:.6e}   stable = {fit.stable}")
    print(f"  poles: {np.array2string(fit.poles, precision=4, separator=', ')}")
    m = fit.metrics
    print(f"  response |R-Z'|/|Z'| : median={m['R_median']:.3%}  "
          f"p90={m['R_p90']:.3%}  real-axis median={m['R_axis_median']:.3%}")
    print(f"  closure  |da|/|alpha|: median={m['A_rel_median']:.3%}  "
          f"p90={m['A_rel_p90']:.3%}  (rms abs {m['A_rms']:.3g})")


def main(Ns: tuple[int, ...] = (3, 4, 5, 6), target: str = "alpha",
         weight: str = "relative") -> None:
    """Refit and print coefficients for pasting into _CACHED."""
    print(f"Fitting Hunana-style closures on Re{NN_RE_RANGE} x Im{NN_IM_RANGE}"
          f"  (target={target}, weight={weight})")
    out = {}
    for N in Ns:
        fit = fit_coefficients(N, target=target, weight=weight)
        _print_report(fit)
        out[N] = fit.a

    print("\n\n=== baselines vs optimized ===")
    entries = {"HP N=3": pade_coefficients(3),
               "Pade N=4": pade_coefficients(4),
               "AAA N=4": aaa_coefficients_n4(),
               "Pade N=5": pade_coefficients(5),
               "Pade N=6": pade_coefficients(6)}
    entries.update({f"opt N={N}": a for N, a in out.items()})
    print(metrics_table(entries))

    print("\n_CACHED = {")
    for N, a in out.items():
        entries_s = ",\n                 ".join(
            f"{c.real:+.10f}{c.imag:+.10f}j" for c in a)
        print(f"    {N}: np.array([{entries_s}]),")
    print("}")


if __name__ == "__main__":
    main()
