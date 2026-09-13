"""Retrain with lambda = 1 (plain and joint-state Gram), unpinned."""
from bot.closures import homog_nn as H
for N in (3, 4):
    for ext in (False, True):
        tag = "_ext" if ext else ""
        p = H.RUN_DIR / f"homog_nn_n{N}{tag}_lam1.eqx"
        if p.exists():
            print("exists:", p); continue
        print(f"=== N={N} ext={ext} lam=1", flush=True)
        H.save(H.train(N, lam=1.0, ext=ext, log_every=20000), N, path=p)
        print("saved", p, flush=True)
