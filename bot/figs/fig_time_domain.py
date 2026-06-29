"""Driver: |E(t)| under Langmuir-projected and generic delta-E ICs.

Generates bot/figures/fig_time_domain.png.
Requires bot/runs/time_domain.npz; auto-generated via run_comparison() if missing.
"""

from bot.figures import fig_time_domain

if __name__ == "__main__":
    fig_time_domain()
