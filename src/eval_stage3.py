# -*- coding: utf-8 -*-
"""阶段三最终分析：强 baseline 复现 + PGF-CLB 混合集成 vs 锁定主结果。

E19 成员 = pgf_nocsi {s0,s1,s2}(固定) + {s0..s2}x{f0..f4}(15 折) + s0_pt  (19)
CLB 固定 = pgf_nocsi_clb {s0,s1,s2}                                    (3)
CLB 折   = pgf_nocsi_clb {s0..s2}x{f0..f4}，按已存在的文件动态收集（链扩员用）
评估（等权、预先注册，不在 test 上选权重）：
  E19 / E8_clb = CLB固定+3+s0折5 / E22 / E27 = E19+8CLB（与 stage3 v1 对齐）
  E{n}_clb = CLB固定+全部已收集折（纯 CLB 全网格）
  E{19+n}  = E19 成员 + 全部 CLB 成员（扩展混合，§8(a) R²>=0.85 判定对象）
配对 t（逐样本 squared error）：扩展 vs E19 / vs E27 / 纯CLB全网格 vs E8_clb 等。
输出 results/stage3_summary.json。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def load_preds(name):
    p = os.path.join(C.RESULTS_DIR, f"preds_{name}.npy")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return np.load(p).astype(np.float64)


def try_load(name):
    p = os.path.join(C.RESULTS_DIR, f"preds_{name}.npy")
    return np.load(p).astype(np.float64) if os.path.exists(p) else None


def paired_t(e1, e2, y):
    d = (e1 - y) ** 2 - (e2 - y) ** 2
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))


def main():
    te = C.get_split("test")
    at = te["anchor_time"]
    y = te["pv"][te["tgt"]].astype(np.float64)
    raw = te["pv"][te["win"]][:, -1].astype(np.float64)
    raw_rmse = float(np.sqrt(np.mean((raw - y) ** 2)))

    e19_members = ([load_preds(f"pgf_nocsi_s{k}") for k in range(3)]
                   + [load_preds(f"pgf_nocsi_s{s}_f{k}") for s in range(3) for k in range(5)]
                   + [load_preds("pgf_nocsi_s0_pt")])
    clb_fixed = [load_preds(f"pgf_nocsi_clb_s{k}") for k in range(3)]
    for k in (3, 4):  # 扩 seed 的固定成员（存在才收）
        m = try_load(f"pgf_nocsi_clb_s{k}")
        if m is not None:
            clb_fixed.append(m)
    clb_folds_s0 = [load_preds(f"pgf_nocsi_clb_s0_f{k}") for k in range(5)]
    conv_ens = load_preds("convlstm_ensemble")
    clsun_ens = load_preds("cl_sunset_ensemble")

    # 动态收集 s1..s4 折（扩链进行中亦可运行，但终局判定需全部 15 折到位）
    clb_folds_extra, missing = [], []
    for s in (1, 2, 3, 4):
        for k in range(5):
            m = try_load(f"pgf_nocsi_clb_s{s}_f{k}")
            if m is None:
                missing.append(f"s{s}_f{k}")
            else:
                clb_folds_extra.append(m)
    clb_all = clb_fixed + clb_folds_s0 + clb_folds_extra

    def ens(tag, members, save=None):
        e = np.mean(members, axis=0)
        r = C.evaluate(y, e, at, tag=tag)
        r["n_members"] = len(members)
        r["skill_vs_raw_overall"] = C.forecast_skill(r["overall"]["rmse"], raw_rmse)
        r["criterion_8a"] = {"rmse_le_2.86": bool(r["overall"]["rmse"] <= 2.86),
                             "r2_ge_0.85": bool(r["overall"]["r2"] >= 0.85)}
        if save:
            np.save(os.path.join(C.RESULTS_DIR, f"preds_{save}.npy"), e)
        return r, e

    rows, preds = {}, {}
    rows["E19_locked_main"], preds["E19"] = ens("E19_locked_main", e19_members)
    rows["E8_clb"], preds["E8_clb"] = ens("E8_clb", clb_fixed + clb_folds_s0, save="pgf_E8_clb")
    rows["E22"], preds["E22"] = ens("E22", e19_members + clb_fixed, save="pgf_E22_mix")
    rows["E27"], preds["E27"] = ens("E27", e19_members + clb_fixed + clb_folds_s0,
                                    save="pgf_E27_mix")
    n_clb = len(clb_all)
    tag_full = f"E{n_clb}_clb"
    tag_ext = f"E{19 + n_clb}"
    rows[tag_full], preds[tag_full] = ens(tag_full, clb_all, save=f"pgf_{tag_full}")
    rows[tag_ext], preds[tag_ext] = ens(tag_ext, e19_members + clb_all, save=f"pgf_{tag_ext}")
    rows["convlstm_ens_ref"], preds["conv"] = ens("convlstm_ens_ref", [conv_ens])
    rows["cl_sunset_ens_ref"], preds["cls"] = ens("cl_sunset_ens_ref", [clsun_ens])

    # CLB 单模型清单
    clb_singles = {}
    for nm, m in ([("s0", clb_fixed[0]), ("s1", clb_fixed[1]), ("s2", clb_fixed[2])]
                  + [(f"f{k}", m) for k, m in enumerate(clb_folds_s0)]):
        clb_singles[nm] = float(np.sqrt(np.mean((m - y) ** 2)))
    for k in (3, 4):
        m = try_load(f"pgf_nocsi_clb_s{k}")
        if m is not None:
            clb_singles[f"s{k}"] = float(np.sqrt(np.mean((m - y) ** 2)))
    for s in (1, 2, 3, 4):
        for k in range(5):
            m = try_load(f"pgf_nocsi_clb_s{s}_f{k}")
            if m is not None:
                clb_singles[f"s{s}f{k}"] = float(np.sqrt(np.mean((m - y) ** 2)))
    vals = list(clb_singles.values())

    tests = {k: paired_t(preds[a], preds[b], y) for k, (a, b) in {
        f"{tag_ext}_vs_E19": (tag_ext, "E19"),
        f"{tag_ext}_vs_E27": (tag_ext, "E27"),
        f"{tag_ext}_vs_{tag_full}": (tag_ext, tag_full),
        f"{tag_full}_vs_E8_clb": (tag_full, "E8_clb"),
        f"{tag_full}_vs_E19": (tag_full, "E19"),
        "E27_vs_E19": ("E27", "E19"),
        f"{tag_ext}_vs_convlstm_ens": (tag_ext, "conv"),
        f"{tag_ext}_vs_cl_sunset_ens": (tag_ext, "cls"),
    }.items() if a != b}

    out = {"missing_clb_folds": missing,
           "n_clb_members": n_clb,
           "clb_single_rmse": clb_singles,
           "clb_single_mean_std": [float(np.mean(vals)), float(np.std(vals, ddof=1))],
           "paired_tests_t": tests,  # 负值 = 左项更优
           "ensembles": rows}
    C.dump_json(out, os.path.join(C.RESULTS_DIR, "stage3_summary.json"))
    if missing:
        print(f"WARNING: {len(missing)} CLB folds missing (grid incomplete): {missing}")
    print(f"CLB singles (n={len(vals)}): mean±std = {out['clb_single_mean_std'][0]:.4f}"
          f"±{out['clb_single_mean_std'][1]:.4f}")
    for t, r in rows.items():
        o, s, c = r["overall"], r["sunny"], r["cloudy"]
        cr = r["criterion_8a"]
        print(f"{t:20s} n={r['n_members']:2d}  overall RMSE={o['rmse']:.4f} R2={o['r2']:.4f} "
              f"sunny={s['rmse']:.4f} cloudy={c['rmse']:.4f} FS_raw={r['skill_vs_raw_overall']:+.1%} "
              f"8a[rmse<2.86:{cr['rmse_le_2.86']} r2>=0.85:{cr['r2_ge_0.85']}]")
    for k, v in tests.items():
        print(f"paired t {k}: {v:.2f}")


if __name__ == "__main__":
    main()
