"""Train origin-pinning ablations: pin='opt' (learned Pade) and pin='none' (free bias)."""
from bot.closures import homog_nn as H
for N in (3, 4):
    for pin in ("opt", "none"):
        p = H.RUN_DIR / f"homog_nn_n{N}_pin{pin}.eqx"
        if p.exists():
            print("exists:", p); continue
        print(f"=== N={N} pin={pin}", flush=True)
        H.save(H.train(N, pin=pin, log_every=20000), N, path=p)
        print("saved", p, flush=True)
