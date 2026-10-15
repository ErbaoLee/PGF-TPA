# -*- coding: utf-8 -*-
"""爬坡事件评估（预注册 2026-10-01）：ramp 子集上 E37 vs 全部 baseline + 叙事三证据。

Ramp 定义（主）：|PV(t+15) - PV(t)| > 3 kW = 10% 额定容量(30kW)；敏感性 15%/20%。
上行/下行分开统计。test n=11180，10% 档事件 ~1815。
证据链：
  (1) 天空图像在爬坡上有正增益：noimg vs 带图像（同 seed）在 ramp 子集的差距
      >> 非爬坡子集；图像-only baseline（ConvLSTM/GRU 对比）同样只在 ramp 上增益。
  (2) 图像模型晴天不行：ConvLSTM-ens/SUNSET-ens 晴天 RMSE >> GRU/E37。
  (3) 动态融合：成员门控 g 在 ramp（尤其上行）显著升高；分支单独误差 vs 融合误差。
输出 results/ramp_summary.json。
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
TH_MAIN, TH_SENS = 3.0, (4.5, 6.0)


def rmse(p, y, m=None):
    m = np.ones(len(y), bool) if m is None else m
    return float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def paired_t(p1, p2, y, m):
    d = (p1[m] - y[m]) ** 2 - (p2[m] - y[m]) ** 2
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))


def main():
    te = C.get_split("test")
    at = te["anchor_time"]
    y = te["pv"][te["tgt"]].astype(np.float64)
    p0 = te["pv"][te["win"]][:, -1].astype(np.float64)
    dP = y - p0
    L = lambda n: np.load(os.path.join(C.RESULTS_DIR, f"preds_{n}.npy")).astype(np.float64)
    mean3 = lambda pre: np.mean([L(f"{pre}_s{k}") for k in range(3)], axis=0)

    masks = {"ramp10": np.abs(dP) > TH_MAIN, "up10": dP > TH_MAIN, "down10": dP < -TH_MAIN,
             "ramp15": np.abs(dP) > TH_SENS[0], "ramp20": np.abs(dP) > TH_SENS[1]}
    m10 = masks["ramp10"]

    models = {
        "OURS_E37": L("pgf_E37"), "E19": L("pgf_E19_all"),
        "ConvLSTM_ens": L("convlstm_ensemble"), "CL_SUNSET_ens": L("cl_sunset_ensemble"),
        "SUNSET_ens": mean3("sunset"), "GRU_ens": mean3("gru"), "MLP_ens": mean3("mlp"),
        "RAW_persistence": p0, "CSI_persistence": L("csi_persistence"),
        # 叙事消融（单 seed 对）
        "PGF_img_s0": L("pgf_nocsi_s0"), "PGF_noimg_s0": L("pgf_noimg_s0"),
        "PGF_nogate_s0": L("pgf_nogate_s0"),
        "PGF_CLB_s0": L("pgf_nocsi_clb_s0"),
    }

    sun, cld = C.sunny_cloudy_masks(at)
    rows = {}
    for k, p in models.items():
        r = {"overall": rmse(p, y), "ramp10": rmse(p, y, m10),
             "nonramp10": rmse(p, y, ~m10), "up10": rmse(p, y, masks["up10"]),
             "down10": rmse(p, y, masks["down10"]), "ramp15": rmse(p, y, masks["ramp15"]),
             "ramp20": rmse(p, y, masks["ramp20"]),
             "sunny": rmse(p, y, sun), "cloudy": rmse(p, y, cld)}
        r["skill_on_ramp10_vs_RAW"] = float(1 - r["ramp10"] / rmse(p0, y, m10)) \
            if k != "RAW_persistence" else 0.0
        rows[k] = r

    tests = {f"E37_vs_{k}": paired_t(models["OURS_E37"], p, y, m10)
             for k, p in models.items() if k not in ("OURS_E37", "RAW_persistence")}
    tests["E37_vs_RAW_per_ramp"] = paired_t(models["OURS_E37"], p0, y, m10)

    # ---------- 门控证据：4 个固定 CLB ckpt 的 g / 分支输出 ----------
    print("extracting gate traces ...")
    dl = DataLoader(C.ForecastDataset("test", None, stride=1, csi=True),
                    batch_size=128, shuffle=False, num_workers=2, pin_memory=True,
                    persistent_workers=True, prefetch_factor=4)
    gs, fa, fb, ff = [], [], [], []
    for name in [f"pgf_nocsi_clb_s{k}" for k in range(4)]:
        model = PGF(use_csi=False, use_cl_b=True).to(DEV)
        model.load_state_dict(torch.load(
            os.path.join(C.RESULTS_DIR, f"{name}_ckpt.pt"), map_location=DEV,
            weights_only=False))
        model.eval()
        gg, aa, bb, ffs = [], [], [], []
        with torch.no_grad():
            for x_img, pv_log, csi_log, pt_log, pt_next, yy in dl:
                f, g, f_a, f_b = model(x_img.to(DEV), pv_log.to(DEV), csi_log.to(DEV),
                                       pt_next.to(DEV), return_parts=True)
                gg.append(g.cpu().numpy()); aa.append(f_a.cpu().numpy())
                bb.append(f_b.cpu().numpy()); ffs.append(f.cpu().numpy())
        gs.append(np.concatenate(gg)); fa.append(np.concatenate(aa))
        fb.append(np.concatenate(bb)); ff.append(np.concatenate(ffs))
        del model
    g = np.mean(gs, axis=0)  # 成员平均门控轨迹（分支输出为单模型证据，取 s0）
    gate_stats = {
        "g_mean_sunny_nonramp": float(g[sun & ~m10].mean()),
        "g_mean_cloudy_nonramp": float(g[cld & ~m10].mean()),
        "g_mean_ramp10": float(g[m10].mean()), "g_mean_up10": float(g[masks["up10"]].mean()),
        "g_mean_down10": float(g[masks["down10"]].mean()),
        "g_mean_nonramp10": float(g[~m10].mean()),
        "spearman_g_absdP": float(_spearman(g, np.abs(dP))),
        # 分支单独 vs 融合（成员 s0）：图像分支只在爬坡上赢时序分支
        "branchA_ramp10": rmse(fa[0], y, m10), "branchB_ramp10": rmse(fb[0], y, m10),
        "fused_ramp10": rmse(ff[0], y, m10),
        "branchA_nonramp": rmse(fa[0], y, ~m10), "branchB_nonramp": rmse(fb[0], y, ~m10),
        "fused_nonramp": rmse(ff[0], y, ~m10),
    }

    out = {"thresholds_kW": {"main": TH_MAIN, "sens": TH_SENS},
           "n_events": {k: int(v.sum()) for k, v in masks.items()},
           "models": rows, "paired_t_on_ramp10": tests, "gate": gate_stats}
    C.dump_json(out, os.path.join(C.RESULTS_DIR, "ramp_summary.json"))

    print(f"\nRAMP (|dP|>3kW, n={m10.sum()})   [skill vs RAW-per on ramp]")
    for k in ["RAW_persistence", "CSI_persistence", "MLP_ens", "GRU_ens", "SUNSET_ens",
              "CL_SUNSET_ens", "ConvLSTM_ens", "E19", "OURS_E37"]:
        r = rows[k]
        print(f"{k:18s} ramp10={r['ramp10']:.4f} up={r['up10']:.4f} down={r['down10']:.4f} "
              f"nonramp={r['nonramp10']:.4f} skill={r['skill_on_ramp10_vs_RAW']:+.1%}")
    print("\npaired t on ramp10 (neg = E37 better):")
    for k, v in tests.items():
        print(f"  {k}: {v:.2f}")
    print("\ngate:", {k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in gate_stats.items()})
    print("\nimage-gain decomposition (RMSE):")
    print(f"  PGF_img_s0:  ramp={rows['PGF_img_s0']['ramp10']:.4f} "
          f"nonramp={rows['PGF_img_s0']['nonramp10']:.4f}")
    print(f"  PGF_noimg_s0 ramp={rows['PGF_noimg_s0']['ramp10']:.4f} "
          f"nonramp={rows['PGF_noimg_s0']['nonramp10']:.4f}")
    print(f"  GRU_ens:     ramp={rows['GRU_ens']['ramp10']:.4f} "
          f"nonramp={rows['GRU_ens']['nonramp10']:.4f}")
    print(f"  ConvLSTM_ens ramp={rows['ConvLSTM_ens']['ramp10']:.4f} "
          f"nonramp={rows['ConvLSTM_ens']['nonramp10']:.4f} "
          f"sunny={rows['ConvLSTM_ens'].get('sunny', float('nan')):.4f}")


if __name__ == "__main__":
    main()
