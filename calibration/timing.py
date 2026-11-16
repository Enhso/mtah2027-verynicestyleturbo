"""Run time of predict and of its stages on external matches."""
import os, sys, time
import numpy as np
from possession import decode, evidence, features, config as cfg
from possession.match import load, CACHE_DIR, path_for
from possession.params import Params
from possession.predict import predict
from harness.score import score

EXT = os.path.join(CACHE_DIR, "ext")
print("MTAH_DATA_DIR set:", bool(os.environ.get("MTAH_DATA_DIR")), "cache dir:", CACHE_DIR,
      "BAR-ATH cache exists:", os.path.exists(path_for("BAR-ATH")))
for name in sys.argv[1:] or ["metrica_game1", "dfl_J03WMX"]:
    t = time.time(); m = load(name, EXT); t_load = time.time() - t
    p = Params()
    t = time.time(); codes = predict(m, p); t_pred = time.time() - t
    tf = te = tv = 0.0
    for a, b in m.periods.tolist():
        t = time.time(); f = features.compute(m.ball[a:b], m.players[a:b], m.speed[a:b], m.team, m.pitch, m.fps, p); tf += time.time() - t
        t = time.time(); e = evidence.evidence(f, [], p); te += time.time() - t
        t = time.time(); decode.viterbi(e, decode.transition_costs(p)); tv += time.time() - t
    t = time.time(); r = score(m, codes); ts = time.time() - t
    print(f"{name}: frames {m.n_frames} slots {m.players.shape[1]} load {t_load:.1f}s predict {t_pred:.1f}s "
          f"(features {tf:.1f} evidence {te:.1f} viterbi {tv:.1f}) score {ts:.1f}s "
          f"spellF1 {r['spell_f1_mean']:.3f} macroF1 {r['macro_f1']:.3f} n_pred {r['n_pred_spells']}")
