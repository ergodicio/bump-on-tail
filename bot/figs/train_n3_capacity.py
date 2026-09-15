"""N=3 capacity/schedule variants, joint-state Gram, lam=1, unpinned, two-mode data."""
import sys
from bot.closures import homog_nn as H
variants = {
    "wide":   dict(hidden=(192, 192, 192)),
    "long":   dict(n_steps=150_000),
    "widelong": dict(hidden=(192, 192, 192), n_steps=150_000, n_samples=400_000),
}
for name in (sys.argv[1:] or list(variants)):
    p = H.RUN_DIR / f"homog_nn_n3_{name}.eqx"
    if p.exists():
        print("exists:", p); continue
    kw = dict(variants[name])
    print(f"=== N=3 {name}: {kw}", flush=True)
    H.save(H.train(3, log_every=25_000, **kw), 3, path=p); print("saved", p, flush=True)
