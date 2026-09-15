"""Modes-only training (no impulse-response states) in the CURRENT configuration:
joint-state Gram, lam = 1, unpinned, two-mode data at both N."""
from bot.closures import homog_nn as H
for N in (3, 4):
    p = H.RUN_DIR / f"homog_nn_n{N}_noimp_current.eqx"
    if p.exists():
        print("exists:", p); continue
    print(f"=== N={N} frac_impulse=0", flush=True)
    H.save(H.train(N, frac_impulse=0.0, log_every=20000), N, path=p)
    print("saved", p, flush=True)
