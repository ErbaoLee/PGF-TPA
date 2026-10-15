# -*- coding: utf-8 -*-
"""爬坡感知迭代终局评估（预注册判据，单次 test 评估）。

E_RAMP = E37 的 37 成员（v2.1: 3固定+15折+pt；CLB: 3固定+15折）+ 6 个 rw 成员
（CNN-B/CLB-B × s0-2，--ramp-w 4）。另报 E_rw6 纯族。
成功判据（预注册）：E_RAMP ramp10 配对 t vs ConvLSTM_ens < -1.96 且 overall ≤ 2.87。
同时提取 rw 成员门控轨迹（g 分组均值 + Spearman(g, |ΔP|)，预期转正）。
输出 results/rw_summary.json。
"""
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from models import PGF  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TH = 3.0


def rmse(p, y, m=None):
    m = np.ones(len(y), bool) if m is None else m
    return float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))


def paired_t(p1, p2, y, m):
    d = (p1[m] - y[m]) ** 2 - (p2[m] - y[m]) ** 2
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    te = C.get_split("test")
    at = te["anchor_time"]
    y = te["pv"][te["tgt"]].astype(np.float64)
    p0 = te["pv"][te["win"]][:, -1].astype(np.float64)
    dP = y - p0
    m10, up, dn = np.abs(dP) > TH, dP > TH, dP < -TH
    sun, cld = C.sunny_cloudy_masks(at)
    L = lambda n: np.load(os.path.join(C.RESULTS_DIR, f"preds_{n}.npy")).astype(np.float64)

    e37_members = ([L(f"pgf_nocsi_s{k}") for k in range(3)]
                   + [L(f"pgf_nocsi_s{s}_f{k}") for s in range(3) for k in range(5)]
                   + [L("pgf_nocsi_s0_pt")]
                   + [L(f"pgf_nocsi_clb_s{k}") for k in range(3)]
                   + [L(f"pgf_nocsi_clb_s{s}_f{k}") for s in range(3) for k in range(5)])
    rw_names = ([f"pgf_nocsi_rw4_s{s}" for s in range(3)]
                + [f"pgf_nocsi_clb_rw4_s{s}" for s in range(3)])
    rw_members = [L(n) for n in rw_names]
    e_ramp = np.mean(e37_members + rw_members, axis=0)
    e_rw6 = np.mean(rw_members, axis=0)
    np.save(os.path.join(C.RESULTS_DIR, "preds_pgf_E_RAMP.npy"), e_ramp)

    conv = L("convlstm_ensemble")
    e37 = L("pgf_E37")
    rows = {}
    for tag, p in {"E_rw6": e_rw6, "E_RAMP": e_ramp, "E37": e37, "ConvLSTM_ens": conv}.items():
        rows[tag] = {"overall": rmse(p, y), "ramp10": rmse(p, y, m10),
                     "up10": rmse(p, y, up), "down10": rmse(p, y, dn),
                     "nonramp10": rmse(p, y, ~m10), "sunny": rmse(p, y, sun),
                     "cloudy": rmse(p, y, cld)}
    tests = {
        "E_RAMP_vs_ConvLSTM_ramp": paired_t(e_ramp, conv, y, m10),
        "E_RAMP_vs_E37_ramp": paired_t(e_ramp, e37, y, m10),
        "E_RAMP_vs_E37_overall": paired_t(e_ramp, e37, y, np.ones(len(y), bool)),
        "E_RAMP_vs_ConvLSTM_up": paired_t(e_ramp, conv, y, up),
        "E_RAMP_vs_ConvLSTM_down": paired_t(e_ramp, conv, y, dn),
        "E_RAMP_vs_E_rw6_ramp": paired_t(e_ramp, e_rw6, y, m10),
    }
    verdict = {"ramp_sig_beats_convlstm": bool(tests["E_RAMP_vs_ConvLSTM_ramp"] < -1.96),
               "overall_le_2.87": bool(rows["E_RAMP"]["overall"] <= 2.87),
               "PASS": bool(tests["E_RAMP_vs_ConvLSTM_ramp"] < -1.96
                            and rows["E_RAMP"]["overall"] <= 2.87)}

    # rw 成员门控轨迹（与 ramp_eval 的 E37 成员门控对照）
    print("extracting rw gate traces ...")
    dl = DataLoader(C.ForecastDataset("test", None, stride=1, csi=True),
                    batch_size=128, shuffle=False, num_workers=2, pin_memory=True,
                    persistent_workers=True, prefetch_factor=4)
    gs = []
    for name, clb in [(n, False) for n in rw_names[:3]] + [(n, True) for n in rw_names[3:]]:
        model = PGF(use_csi=False, use_cl_b=clb).to(DEV)
        model.load_state_dict(torch.load(
            os.path.join(C.RESULTS_DIR, f"{name}_ckpt.pt"), map_location=DEV,
            weights_only=False))
        model.eval()
        gg = []
        with torch.no_grad():
            for x_img, pv_log, csi_log, pt_log, pt_next, yy in dl:
                _, g, _, _ = model(x_img.to(DEV), pv_log.to(DEV), csi_log.to(DEV),
                                   pt_next.to(DEV), return_parts=True)
                gg.append(g.cpu().numpy())
        gs.append(np.concatenate(gg))
        del model
    g = np.mean(gs, axis=0)
    gate = {"g_mean_ramp10": float(g[m10].mean()), "g_mean_up10": float(g[up].mean()),
            "g_mean_down10": float(g[dn].mean()), "g_mean_nonramp10": float(g[~m10].mean()),
            "g_mean_sunny_nonramp": float(g[sun & ~m10].mean()),
            "spearman_g_absdP": _spearman(g, np.abs(dP)),
            "对照_E37成员": {"g_mean_ramp10": 0.6646, "g_mean_nonramp10": 0.6612,
                              "spearman": -0.3398}}

    out = {"rows": rows, "paired_t": tests, "verdict": verdict, "gate_rw": gate,
           "rw_members": rw_names}
    C.dump_json(out, os.path.join(C.RESULTS_DIR, "rw_summary.json"))
    for t, r in rows.items():
        print(f"{t:14s} overall={r['overall']:.4f} ramp10={r['ramp10']:.4f} "
              f"up={r['up10']:.4f} down={r['down10']:.4f} nonramp={r['nonramp10']:.4f} "
              f"sunny={r['sunny']:.4f}")
    for k, v in tests.items():
        print(f"paired t {k}: {v:.2f}")
    print("VERDICT:", verdict)
    print("gate_rw:", {k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in gate.items()})


if __name__ == "__main__":
    main()
