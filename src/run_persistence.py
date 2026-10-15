# -*- coding: utf-8 -*-
"""零参数参照：RAW persistence 与官方 CSI persistence（锁定 test）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

te = C.get_split("test")
at = te["anchor_time"]
y = te["pv"][te["tgt"]].astype(np.float64)
pv_log = te["pv"][te["win"]].astype(np.float64)

res = {"model": "persistence", "seed": 0}
for name, pred in [
    ("raw_persistence", pv_log[:, -1]),
    ("csi_persistence", C.csi_persistence(at, pv_log, y)),
]:
    r = C.evaluate(y, pred, at, tag=name)
    res[name] = r
    print(name, "overall:", {k: round(v, 4) for k, v in r["overall"].items()})
    print(name, "sunny:", {k: round(v, 4) for k, v in r["sunny"].items() if k != 'n'})
    print(name, "cloudy:", {k: round(v, 4) for k, v in r["cloudy"].items() if k != 'n'})
    np.save(os.path.join(C.RESULTS_DIR, f"preds_{name}.npy"), pred)

C.dump_json(res, os.path.join(C.RESULTS_DIR, "persistence.json"))
