"""Standing-wave figure with the modes-only (no impulse-response) closures
(plain Gram, lam = 0.3, asym pin -- the configuration of that ablation)."""
import bot.figs.fig_standing_wave as F
from bot.closures.homog_nn import RUN_DIR
F.HOMOG_CKPT = {3: RUN_DIR / "homog_nn_n3_noimp.eqx", 4: RUN_DIR / "homog_nn_n4_noimp.eqx"}
F.HOMOG_KW = dict(pin="asym", ext=False); F.HOMOG_LAM = 0.3
F.HOMOG_LABEL = r"NN homog., no impulse"
F.SAVE_PNG = "bot/figures/homog/fig_standing_wave_noimp.png"
F.main()
