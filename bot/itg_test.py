"""ITG marginal-stability test: slab geometry, zero FLR.

Compares the exact kinetic ITG dispersion (Hammett-Perkins PRL 64 3019 1990,
Eq. 12) with fluid closures including our BoT-trained NN closures:

  1. HP 3-moment correct  (Γ=3, μ₁=0, χ₁=2/√π)        — HP recommended
  2. HP 3-moment Brag     (Γ=5/3, μ₁=0, χ₁=2/√π)       — Braginskii-like
  3. BoT Padé N=3         (from closures/pade.py)         — Taylor-matched rational
  4. U2 exact             (beta_exact = ζ²−1/Z′)          — should equal kinetic
  5. U3 exact             (alpha_exact = ζ³−ζ/Z′)         — should equal kinetic
  6. BoT U2 NN            (from runs/nn_u2_r1.eqx)        — neural beta closure
  7. BoT N=4 NN           (from runs/n4_nn.eqx)           — neural alpha4 closure

Key result (HP 1990): the kinetic threshold is η_i = 2.  The HP closure
reproduces it exactly.  The NN closures are highly accurate in Z(ζ) (errors
O(1e-3)) because they were trained over the complex ξ_b plane, unlike Padé
which is Taylor-matched at ξ=0 and has O(1) error at the complex ITG roots.

Normalization
─────────────
Frequencies in units of |k_∥| v_{ti} √2:
    ζ   = ω / (|k_∥| v_{ti} √2)
    ζ_* = ω_* / (|k_∥| v_{ti} √2)
    η_i = L_n / L_T ,    τ = T_i / T_e

The Maxwellian is f_0 ∝ exp(−v_∥²/(2v_{ti}²)).  In BoT notation (v_b=1,
u_b=0) ξ_b = ω/k = ζ, so the closure coefficients apply directly.

NN Z-function derivation
─────────────────────────
All exact closures give U_0 = Z'(ζ) on the eigenmode manifold (trivially
reproducing kinetic). For NN approximations:

  U2 NN: beta_nn(ζ) ≈ ζ²−1/Z'(ζ)  →  U_0 = 1/(ζ²−beta_nn)
  N=4 NN: alpha4_nn(r1,r2,r3) ≈ ζ⁴−(ζ²+3/2)/Z'(ζ)  →  U_0 = (ζ²+3/2)/(ζ⁴−alpha4_nn)

Then Z_nn(ζ) = (−U_0/2 − 1)/ζ  [from Z'=−2(1+ζZ) → Z=(−Z'/2−1)/ζ].
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import root
from scipy.special import wofz

from bot.closures.pade import fluid_U0, pade_coefficients

SQRT_PI = float(np.sqrt(np.pi))


# ── plasma dispersion function ────────────────────────────────────────────────

def Z(zeta: complex) -> complex:
    """Z(ζ) = i√π w(ζ) via Faddeeva."""
    z = complex(zeta)
    if np.isnan(z.real) or np.isnan(z.imag):
        return complex(float("nan"), float("nan"))
    return 1j * SQRT_PI * wofz(z)


# ── general HP 3-moment R_3 and derived Z_3 ──────────────────────────────────

def R3_hp(zeta: complex, Gamma: float = 3.0, mu1: float = 0.0,
          chi1: float = 2.0 / SQRT_PI) -> complex:
    """HP 3-moment 1D response function R_3(ζ)  (HP 1990 Eq. 9).

    HP recommended: Γ=3, μ₁=0, χ₁=2/√π.
    Braginskii-like:  Γ=5/3, μ₁=0, χ₁=2/√π  (yields wrong ITG threshold).
    """
    z = complex(zeta)
    num = chi1 - 1j * z
    den = (chi1
           - 1j * Gamma * z
           - 2j * chi1 * mu1 * z
           - 2.0 * chi1 * z**2
           - 2.0 * mu1 * z**2
           + 2j * z**3)
    return num / den


def Z3(zeta: complex, **kw) -> complex:
    """3-pole approximation to Z(ζ) from R_3: Z_3 = (R_3 − 1) / ζ."""
    z = complex(zeta)
    if abs(z) < 1e-14:
        return 1j * SQRT_PI  # Z(0) = i√π exact
    return (R3_hp(z, **kw) - 1.0) / z


# ── BoT Padé N=3 converted to a Z approximation ──────────────────────────────

_PADE3 = None


def _get_pade3():
    global _PADE3
    if _PADE3 is None:
        _PADE3 = pade_coefficients(3)
    return _PADE3


def Z_pade3(zeta: complex) -> complex:
    """BoT Padé N=3 approximation to Z(ζ).

    Identity: fluid_U0 ≈ Z′(ξ) (closures/pade.py matches the closed fluid
    response to the kinetic Z′ directly), so via Z′=−2(1+ζZ):
        Z(ζ) ≈ (−fluid_U0(ζ, a)/2 − 1) / ζ.

    Valid for the single-Maxwellian 1D problem (ξ_b = ζ when u_b=0, v_b=1).
    """
    z_val = complex(zeta)
    if abs(z_val) < 1e-14:
        return 1j * SQRT_PI
    z_arr = np.asarray([z_val])
    U0 = complex(fluid_U0(z_arr, _get_pade3())[0])
    return _Z_from_U0(U0, z_val)


# ── BoT NN Z approximations ──────────────────────────────────────────────────

_U2_NN_MODEL = None
_N4_NN_MODEL = None


def _get_u2_nn():
    global _U2_NN_MODEL
    if _U2_NN_MODEL is None:
        import equinox as eqx
        import jax
        from bot.closures.u2 import MLP_u2
        arch = MLP_u2(key=jax.random.PRNGKey(0))
        _U2_NN_MODEL = eqx.tree_deserialise_leaves("bot/runs/nn_u2_r1.eqx", arch)
    return _U2_NN_MODEL


def _get_n4_nn():
    global _N4_NN_MODEL
    if _N4_NN_MODEL is None:
        import equinox as eqx
        from bot.closures.n4_nn import MLP_n4, CKPT
        arch = MLP_n4(key=__import__("jax").random.PRNGKey(0))
        _N4_NN_MODEL = eqx.tree_deserialise_leaves(str(CKPT), arch)
    return _N4_NN_MODEL


def _Z_from_U0(U0: complex, z: complex) -> complex:
    """Z(ζ) from U_0 = Z'(ζ): Z = (−U_0/2 − 1) / ζ."""
    return (-U0 / 2.0 - 1.0) / z


def Z_u2_exact(zeta: complex) -> complex:
    """Z from exact U2 closure.

    N=2 fluid system (U_0, U_1) closed by U_2 = beta(ζ) U_0 with the exact
    kinetic beta(ζ) = ζ²−1/Z'(ζ).  Eigenmode condition gives U_0 = 1/(ζ²−beta).
    Reduces to exact kinetic Z (verification of the closure derivation).
    """
    z = complex(zeta)
    if abs(z) < 1e-14:
        return 1j * SQRT_PI
    from bot.closures.u2 import beta as beta_exact
    b = complex(beta_exact(z))
    denom = z**2 - b
    if abs(denom) < 1e-14:
        return complex(float("nan"), float("nan"))
    U0 = 1.0 / denom
    return _Z_from_U0(U0, z)


def Z_u3_exact(zeta: complex) -> complex:
    """Z from exact U3 inference closure.

    N=3 fluid system (U_0, U_1, U_2) closed by U_3 = alpha(ζ) U_0 with the
    exact kinetic alpha(ζ) = ζ³−ζ/Z'(ζ).  Eigenmode condition gives
    U_0 = ζ/(ζ³−alpha).  Reduces to exact kinetic Z (verification).
    """
    z = complex(zeta)
    if abs(z) < 1e-14:
        return 1j * SQRT_PI
    from bot.closures.inference import alpha as alpha_exact
    a = complex(alpha_exact(z))
    denom = z**3 - a
    if abs(denom) < 1e-14:
        return complex(float("nan"), float("nan"))
    U0 = z / denom
    return _Z_from_U0(U0, z)


def Z_u2_nn(zeta: complex) -> complex:
    """Z from U2 NN closure.

    U2 NN learns beta_nn(ζ) ≈ ζ²−1/Z'(ζ).  The 2-moment fluid system gives
    U_0 = 1/(ζ²−beta_nn), and Z = (−U_0/2−1)/ζ.  Input feature is r_1=ζ
    (trivially the eigenmode frequency since r_1=U_1/U_0=ζ on manifold).
    """
    z = complex(zeta)
    if abs(z) < 1e-14:
        return 1j * SQRT_PI
    from bot.closures.u2 import model_beta
    beta_hat = complex(model_beta(_get_u2_nn(), z))
    U0 = 1.0 / (z**2 - beta_hat)
    return _Z_from_U0(U0, z)


def Z_n4_nn(zeta: complex) -> complex:
    """Z from N=4 NN closure.

    N=4 NN learns alpha4_nn(r1,r2,r3) ≈ ζ⁴−(ζ²+3/2)/Z'(ζ).  Features are
    the on-manifold moment ratios (r1=ζ, r2=beta(ζ), r3=alpha(ζ)) evaluated
    with the exact Z'.  The 4-moment fluid system gives U_0=(ζ²+3/2)/(ζ⁴−alpha4_nn).
    """
    z = complex(zeta)
    if abs(z) < 1e-14:
        return 1j * SQRT_PI
    from bot.closures.n4_nn import n4_predict_alpha4
    from bot.closures.inference import moment_ratios_n4
    r1, r2, r3 = moment_ratios_n4(z)
    a4 = complex(n4_predict_alpha4(_get_n4_nn(), r1, r2, r3))
    U0 = (z**2 + 1.5) / (z**4 - a4)
    return _Z_from_U0(U0, z)


# ── slab ITG dispersion  D(ζ) = R_s + τ  (HP Eq. 12) ────────────────────────

def _D_slab(Z_func, zeta: complex, zeta_star: float,
            eta: float, tau: float) -> complex:
    """Slab ITG dispersion function: D = R_s + τ, root → mode frequency.

    R_s = 1 − η ζ_* + [ζ − ζ_*(1 − η/2 + η ζ²)] Z(ζ)   (HP Eq. 12)
    """
    z = complex(zeta)
    Zv = Z_func(z)
    Rs = (1.0 - eta * zeta_star
          + (z - zeta_star * (1.0 - 0.5 * eta + eta * z**2)) * Zv)
    return Rs + tau


def D_kinetic(zeta, zeta_star, eta, tau=1.0):
    return _D_slab(Z, zeta, zeta_star, eta, tau)

def D_hp(zeta, zeta_star, eta, tau=1.0):
    return _D_slab(lambda z: Z3(z, Gamma=3.0, mu1=0.0), zeta, zeta_star, eta, tau)

def D_brag(zeta, zeta_star, eta, tau=1.0):
    """Braginskii-like: Γ=5/3, μ₁=0, χ₁=2/√π — predicts wrong threshold."""
    return _D_slab(lambda z: Z3(z, Gamma=5.0/3.0, mu1=0.0), zeta, zeta_star, eta, tau)

def D_pade(zeta, zeta_star, eta, tau=1.0):
    return _D_slab(Z_pade3, zeta, zeta_star, eta, tau)

def D_u2_exact(zeta, zeta_star, eta, tau=1.0):
    return _D_slab(Z_u2_exact, zeta, zeta_star, eta, tau)

def D_u3_exact(zeta, zeta_star, eta, tau=1.0):
    return _D_slab(Z_u3_exact, zeta, zeta_star, eta, tau)

def D_u2_nn(zeta, zeta_star, eta, tau=1.0):
    return _D_slab(Z_u2_nn, zeta, zeta_star, eta, tau)

def D_n4_nn(zeta, zeta_star, eta, tau=1.0):
    return _D_slab(Z_n4_nn, zeta, zeta_star, eta, tau)


# ── Newton root-finder ────────────────────────────────────────────────────────

def find_mode(D_func, zeta0: complex, zeta_star: float,
              eta: float, tau: float = 1.0) -> complex:
    """Newton-solve D(ζ)=0 near ζ₀; return complex ζ (nan if failed)."""
    def res(x):
        v = D_func(complex(x[0], x[1]), zeta_star, eta, tau)
        return [v.real, v.imag]
    sol = root(res, [zeta0.real, zeta0.imag], method="hybr", tol=1e-12)
    if not sol.success:
        return complex(float("nan"), float("nan"))
    return complex(sol.x[0], sol.x[1])


def _max_growing_root(D_func, zeta_star: float, eta: float, tau: float,
                      re_lo: float = -2.0, re_hi: float = 3.0,
                      im_lo: float = 0.001, im_hi: float = 4.0,
                      n_re: int = 18, n_im: int = 18) -> complex:
    """Find UHP root of D with maximum Im(ζ) via grid search + Newton.

    Searches both Re < 0 (drift-wave branch) and Re > 0 (ITG branch) so the
    correct growing root is found at every η, including near the η=2 threshold
    where the two branches exchange dominance.

    Returns complex(nan, 0) if no root with Im > 0 is found.
    """
    best_z = complex(float("nan"), 0.0)
    best_im = 0.0
    for re_v in np.linspace(re_lo, re_hi, n_re):
        for im_v in np.linspace(im_lo, im_hi, n_im):
            z = find_mode(D_func, complex(re_v, im_v), zeta_star, eta, tau)
            if np.isnan(z.real):
                continue
            if z.imag > best_im and abs(D_func(z, zeta_star, eta, tau)) < 1e-8:
                best_im = z.imag
                best_z = z
    return best_z


# ── η_i scan ──────────────────────────────────────────────────────────────────

def scan_eta(eta_values: np.ndarray, zeta_star: float,
             tau: float = 1.0) -> dict[str, np.ndarray]:
    """Scan η_i: return the most-unstable root at each η for each closure.

    At ζ_*=1, τ=1: two branches coexist — a drift-wave root (Re < 0) dominant
    for η < 2, and an ITG root (Re > 0) dominant for η > 2.  Both have Im=0
    at η=2 (exact marginal stability, all closures).  Grid search over the full
    upper half-plane finds whichever branch is most unstable at each η.

    Returns dict with keys 'kin', 'hp', 'brag', 'pade', 'u2_exact', 'u3_exact',
    'u2_nn', 'n4_nn'.  Each value is a complex array; Im gives growth rate,
    Re gives real frequency.
    """
    # Fast closures: full grid search at each η
    fast_funcs = {
        "kin":      D_kinetic,
        "hp":       D_hp,
        "brag":     D_brag,
        "pade":     D_pade,
        "u2_exact": D_u2_exact,
        "u3_exact": D_u3_exact,
    }
    # NN closures: expensive per call — seed from kinetic roots instead of grid
    nn_funcs = {
        "u2_nn": D_u2_nn,
        "n4_nn": D_n4_nn,
    }
    n = len(eta_values)
    out = {k: np.empty(n, dtype=complex) for k in {**fast_funcs, **nn_funcs}}

    # Step 1: compute fast closures with full grid search
    kin_roots = np.empty(n, dtype=complex)
    for i, eta in enumerate(eta_values):
        for k, D in fast_funcs.items():
            z = _max_growing_root(D, zeta_star, eta, tau)
            out[k][i] = z
            if k == "kin":
                kin_roots[i] = z

    # Step 2: NN closures — seed Newton from kinetic root (NN ≈ kinetic, so
    # roots are close).  A small backup grid is tried if kinetic seed fails.
    print("  Computing NN closures (seeded from kinetic roots)...")
    for k, D in nn_funcs.items():
        z_prev = None  # last successfully converged root, for continuity seeding
        for i, eta in enumerate(eta_values):
            def _accept(z):
                return (not np.isnan(z.real) and z.imag > 0
                        and abs(D(z, zeta_star, eta, tau)) < 1e-6)

            # Seed candidates in priority order: kinetic root, previous-eta
            # NN root (continuity), then a generic fallback point.
            seeds = []
            if not np.isnan(kin_roots[i].real):
                seeds.append(kin_roots[i])
            if z_prev is not None:
                seeds.append(z_prev)
            seeds.append(complex(0.5, 0.3))

            z = complex(float("nan"), 0.0)
            for z_seed in seeds:
                z_try = find_mode(D, z_seed, zeta_star, eta, tau)
                if _accept(z_try):
                    z = z_try
                    break

            if np.isnan(z.real):
                # Backup: small grid search with coarse resolution
                z = _max_growing_root(D, zeta_star, eta, tau, n_re=8, n_im=8)

            out[k][i] = z
            if not np.isnan(z.real):
                z_prev = z

    return out


# ── analytic near-threshold curve ─────────────────────────────────────────────

def threshold_analytic(Gamma: float = 3.0, mu1: float = 0.0) -> float:
    """HP near-threshold formula: η_i = Γ − 1 + 2 χ₁ μ₁."""
    chi1 = 2.0 / SQRT_PI
    return Gamma - 1.0 + 2.0 * chi1 * mu1


def growth_rate_near_threshold(eta_values, Gamma: float = 3.0,
                                mu1: float = 0.0) -> np.ndarray:
    """Im(ζ) = (η_i − η_thresh) / (2(χ₁+μ₁))  (HP Eq. near threshold)."""
    chi1 = 2.0 / SQRT_PI
    eta_th = threshold_analytic(Gamma, mu1)
    return np.asarray([(eta - eta_th) / (2.0 * (chi1 + mu1))
                       for eta in eta_values])


# ── figure ────────────────────────────────────────────────────────────────────

def fig_itg_threshold(zeta_star: float = 1.0, tau: float = 1.0,
                      eta_range: tuple[float, float] = (0.1, 5.0),
                      n_eta: int = 50,
                      save_path: str = "bot/figures/fig_itg_threshold.png"):
    """Growth rate Im(ζ) vs η_i for kinetic and HP fluid closures.

    At ζ_*=1, τ=1: all closures show exact marginal stability at η=2 (HP
    threshold).  Below η=2 a drift-wave mode (Re<0) grows; above η=2 an ITG
    mode (Re>0) grows.  HP fluid better matches the kinetic growth rate than
    the Braginskii-Γ variant.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eta_values = np.linspace(*eta_range, n_eta)
    print(f"Scanning η_i ∈ [{eta_range[0]}, {eta_range[1]}]"
          f"  |  ζ_* = {zeta_star},  τ = {tau}")

    results = scan_eta(eta_values, zeta_star, tau)

    eta_th_hp = 2.0  # marginal at ζ_*=(1+τ)/η_th → η_th=2 for τ=1, ζ_*=1

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── panel 1: growth rate ──────────────────────────────────────────────────
    ax = axes[0]
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(eta_th_hp, color="gray", lw=0.8, ls="--", alpha=0.7,
               label=r"$\eta_{th}=2$ (HP threshold)")

    def gr(key):
        return np.maximum(results[key].imag, 0.0)

    ax.plot(eta_values, gr("kin"),
            "k-",   lw=2.5, label="Kinetic (exact)")
    ax.plot(eta_values, gr("hp"),
            "C1--", lw=2.0, label=r"HP 3-mom ($\Gamma\!=\!3$)")
    ax.plot(eta_values, gr("brag"),
            "C2:",  lw=2.0, label=r"Brag-$\Gamma$ ($\Gamma\!=\!5/3$)")
    ax.plot(eta_values, gr("pade"),
            "C0-.", lw=1.5, label="BoT Padé $N=3$")
    ax.plot(eta_values, gr("u2_exact"),
            "C5-",  lw=1.5, label="U2 exact")
    ax.plot(eta_values, gr("u3_exact"),
            "C6--", lw=1.5, label="U3 exact")
    ax.plot(eta_values, gr("u2_nn"),
            "C3-",  lw=1.8, label="BoT U2 NN")
    ax.plot(eta_values, gr("n4_nn"),
            "C4--", lw=1.8, label="BoT N=4 NN")

    ax.annotate(r"$\eta_{th}=2$", xy=(eta_th_hp, 0),
                xytext=(eta_th_hp + 0.15, 0.15), color="gray", fontsize=10,
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    ax.set_xlabel(r"$\eta_i = L_n / L_T$", fontsize=12)
    ax.set_ylabel(r"$\mathrm{Im}(\zeta)\;=\;\gamma\,/\,(|k_\parallel| v_{ti} \sqrt{2})$",
                  fontsize=11)
    ax.set_title("Growth rate  (max over both branches)", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(*eta_range)

    # ── panel 2: real frequency of most-unstable root ─────────────────────────
    ax = axes[1]
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(eta_th_hp, color="gray", lw=0.8, ls="--", alpha=0.7)

    ax.plot(eta_values, results["kin"].real,      "k-",   lw=2.5, label="Kinetic")
    ax.plot(eta_values, results["hp"].real,       "C1--", lw=2.0, label="HP 3-moment")
    ax.plot(eta_values, results["brag"].real,     "C2:",  lw=2.0, label=r"Brag-$\Gamma$")
    ax.plot(eta_values, results["pade"].real,     "C0-.", lw=1.5, label="BoT Padé N=3")
    ax.plot(eta_values, results["u2_exact"].real, "C5-",  lw=1.5, label="U2 exact")
    ax.plot(eta_values, results["u3_exact"].real, "C6--", lw=1.5, label="U3 exact")
    ax.plot(eta_values, results["u2_nn"].real,    "C3-",  lw=1.8, label="BoT U2 NN")
    ax.plot(eta_values, results["n4_nn"].real,    "C4--", lw=1.8, label="BoT N=4 NN")

    ax.set_xlabel(r"$\eta_i = L_n / L_T$", fontsize=12)
    ax.set_ylabel(r"$\mathrm{Re}(\zeta)\;=\;\omega_r\,/\,(|k_\parallel| v_{ti} \sqrt{2})$",
                  fontsize=11)
    ax.set_title("Real frequency  (most-unstable root)", fontsize=12)
    ax.legend(fontsize=8)
    ax.set_xlim(*eta_range)

    fig.suptitle(
        rf"Slab ITG dispersion  ($\zeta_* = {zeta_star}$,  $\tau = T_i/T_e = {tau}$)"
        "\n"
        r"HP 1990 Eq.12: kinetic vs fluid closures,  $k_\perp\rho_i\to 0$",
        fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")
    return results, eta_values


# ── threshold + growth rate report ────────────────────────────────────────────

def report(results: dict[str, np.ndarray], eta_values: np.ndarray):
    """Print threshold positions and growth rate comparison table."""
    labels = {
        "kin":      "Kinetic (exact)",
        "hp":       "HP 3-moment (Γ=3)",
        "brag":     "Brag-Γ (Γ=5/3)",
        "pade":     "BoT Padé N=3",
        "u2_exact": "U2 exact closure",
        "u3_exact": "U3 exact closure",
        "u2_nn":    "BoT U2 NN",
        "n4_nn":    "BoT N=4 NN",
    }

    print("\n── Marginal stability threshold (Im(ζ) = 0) ──")
    print("  [Analytic: at ζ_*=1, τ=1 all closures → η_th = (1+τ)/ζ_* = 2]")
    for k, name in labels.items():
        arr = results[k]
        im = np.maximum(arr.imag, 0.0)
        # Find crossing from growing → stable (η increasing through threshold)
        # Look for the last zero-crossing going upward through η_th=2
        eta_th_list = []
        for j in range(1, len(im) - 1):  # skip j=0 to avoid scan-boundary artifacts
            if im[j] > 1e-4 > im[j+1] or im[j+1] > 1e-4 > im[j]:
                # linear interpolation: find η where Im crosses 0
                t = im[j] / (im[j] - im[j+1])
                eta_th_list.append(float(eta_values[j] + t * (eta_values[j+1] - eta_values[j])))
        if eta_th_list:
            print(f"  {name:28s}: η_i = " +
                  ", ".join(f"{x:.4f}" for x in eta_th_list))
        else:
            print(f"  {name:28s}: no Im=0 crossing found in scanned range")

    print("\n── Growth rate Im(ζ) at selected η_i ──")
    print(f"  (positive = growing; drift-wave branch for η<2, ITG branch for η>2)")
    header = f"  {'η_i':>5}  " + "".join(f"  {k:>14}" for k in labels)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for eta_check in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        i = np.argmin(np.abs(eta_values - eta_check))
        row = f"  {eta_values[i]:>5.2f}  "
        for k in labels:
            gr = max(results[k][i].imag, 0.0)
            row += f"  {gr:>14.4f}"
        print(row)


# Driver: python -m bot.figs.fig_itg_threshold (see bot/figs/fig_itg_threshold.py)
