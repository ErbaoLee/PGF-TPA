# -*- coding: utf-8 -*-
"""阶段二最终分析：EMA fold×seed 网格 + 全集成 vs 锁定主结果。

成员库：pgf_nocsi {s0,s1,s2}(固定划分) + {s0..s2}×{f0..f4}(15 折网格,
s1/s2 为 EMA 训练) + s0_pt(nowcast 预训练微调)。输出 results/stage2_summary.json。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def load_preds(name):
    p = os.path.join(C.RESULTS_DIR, f"preds_{name}.npy")
    return np.load(p).astype(np.float64) if os.path.exists(p) else None


def paired_t(e1, e2):
    d = e1 - e2
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))


def main():
    te = C.get_split("test")
    at = te["anchor_time"]
    y = te["pv"][te["tgt"]].astype(np.float64)
    raw = te["pv"][te["win"]][:, -1].astype(np.float64)
    raw_rmse = float(np.sqrt(np.mean((raw - y) ** 2)))

    fixed3 = [load_preds(f"pgf_nocsi_s{k}") for k in range(3)]
    folds15 = [load_preds(f"pgf_nocsi_s{s}_f{k}") for s in range(3) for k in range(5)]
    pt = [load_preds("pgf_nocsi_s0_pt")]
    sun3 = [load_preds(f"sunset_s{k}") for k in range(3)]

    def ens(tag, members, save=None):
        ms = [m for m in members if m is not None]
        e = np.mean(ms, axis=0)
        r = C.evaluate(y, e, at, tag=tag)
        r["n_members"] = len(ms)
        r["skill_vs_raw_overall"] = C.forecast_skill(r["overall"]["rmse"], raw_rmse)
        if save:
            np.save(os.path.join(C.RESULTS_DIR, f"preds_{save}.npy"), e)
        return r, (e - y) ** 2

    out = {"raw_rmse": raw_rmse}
    fold_rmse = []
    for s in range(3):
        for k in range(5):
            m = load_preds(f"pgf_nocsi_s{s}_f{k}")
            if m is not None:
                fold_rmse.append(float(np.sqrt(np.mean((m - y) ** 2))))
    out["fold15_single_rmse"] = fold_rmse
    out["fold15_single_mean_std"] = [float(np.mean(fold_rmse)),
                                     float(np.std(fold_rmse, ddof=1))]

    rows, sqs = {}, {}
    rows["pgf_E3_locked_main"], sqs["E3"] = ens("pgf_E3_locked_main", fixed3)
    rows["pgf_E15_fold_grid"], sqs["E15"] = ens("pgf_E15_fold_grid", folds15,
                                                save="pgf_E15_fold_grid")
    rows["pgf_E19_all"], sqs["E19"] = ens("pgf_E19_all", folds15 + fixed3 + pt,
                                          save="pgf_E19_all")
    sun_e = np.mean([m for m in sun3 if m is not None], axis=0)
    sq_sun = (sun_e - y) ** 2

    tests = {"E19_vs_E3": paired_t(sqs["E19"], sqs["E3"]),
             "E15_vs_E3": paired_t(sqs["E15"], sqs["E3"]),
             "E19_vs_sunset_ens": paired_t(sqs["E19"], sq_sun)}
    out["paired_tests_t"] = tests  # 负值 = 左项更优
    out["ensembles"] = rows

    C.dump_json(out, os.path.join(C.RESULTS_DIR, "stage2_summary.json"))
    print(f"RAW persistence overall RMSE = {raw_rmse:.4f}")
    ms = out["fold15_single_mean_std"]
    print(f"fold-grid singles (n={len(fold_rmse)}): mean±std = {ms[0]:.4f}±{ms[1]:.4f}")
    for t, r in rows.items():
        o, s, c = r["overall"], r["sunny"], r["cloudy"]
        print(f"{t:22s} n={r['n_members']:2d}  overall RMSE={o['rmse']:.4f} R2={o['r2']:.4f} "
              f"sunny={s['rmse']:.4f} cloudy={c['rmse']:.4f} FS_raw={r['skill_vs_raw_overall']:+.1%}")
    for k, v in tests.items():
        print(f"paired t {k}: {v:.2f}")


if __name__ == "__main__":
    main()
