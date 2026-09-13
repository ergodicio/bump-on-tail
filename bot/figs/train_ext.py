"""Train the joint-state-Gram (ext) variant, N=3 and N=4."""
from bot.closures import homog_nn as H
for N in (3, 4):
    p = H.RUN_DIR / f"homog_nn_n{N}_ext.eqx"
    if p.exists():
        print("exists:", p); continue
    print(f"=== N={N} ext", flush=True)
    H.save(H.train(N, ext=True, log_every=20000), N, path=p)
    print("saved", p, flush=True)
