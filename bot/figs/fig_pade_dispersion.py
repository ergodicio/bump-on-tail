"""Driver: gamma(k) Pade hierarchy vs kinetic, representative (u_b, eps).

Generates bot/figures/fig_pade_dispersion.png.
No pre-computed data required (runs kinetic_peak + trace_fluid live);
loads bot/runs/inference_mlp.eqx for the inference-closure overlay.
"""

from bot.figures import fig_dispersion_examples

if __name__ == "__main__":
    fig_dispersion_examples()
