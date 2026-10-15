# -*- coding: utf-8 -*-
"""分支级诊断：AWA v2 vs CLB 的 B 分支输出与门控行为（推理级）。

问题：AWA v2 单模仍输 CLB（s0: 2.952 vs 2.904），晴天差距最大
（0.344 vs 0.263）。定位：B 分支单独误差、A 分支单独误差、融合误差、
门控 g 分布（晴天/阴天/ramp）——判断失败在"B 输出质量"还是"门控配合"。
"""
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from models import PGF  # noqa: E402


@torch.no_grad()
def parts_of(ckpt, kw):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = PGF(use_csi=False, **kw).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.eval()
    dl = DataLoader(C.ForecastDataset("test", None, stride=1, csi=True),
                    batch_size=512, num_workers=0)
    fa, fb, gg, ys = [], [], [], []
    for x_img, pv_log, csi_log, pt_log, pt_next, y in dl:
        f, g, f_a, f_b = model(x_img.to(dev), pv_log.to(dev), csi_log.to(dev),
                               pt_next.to(dev), return_parts=True)
        fa.append(f_a.cpu().numpy()); fb.append(f_b.cpu().numpy())
        gg.append(g.cpu().numpy()); ys.append(y.numpy())
    return (np.concatenate(x) for x in (fa, fb, gg, ys))


def rmse(p, y, m):
    return float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))


def main():
    te = C.get_split("test")
    y = te["pv"][te["tgt"]].astype(np.float64)
    p0 = te["pv"][te["win"]][:, -1].astype(np.float64)
    dP = y - p0
    sunny, cloudy = C.sunny_cloudy_masks(te["anchor_time"])
    masks = {"overall": np.ones(len(y), bool), "sunny": sunny, "cloudy": cloudy,
             "ramp": np.abs(dP) > 3.0, "nonramp": np.abs(dP) <= 3.0}

    for name, ck, kw in [
            ("CLB_s0", "results/pgf_nocsi_clb_s0_ckpt.pt", dict(use_cl_b=True)),
            ("AWA2_s0", "results/pgf_nocsi_awa_s0_ckpt.pt", dict(use_awa_b=True))]:
        f_a, f_b, g, _ = parts_of(ck, kw)
        print(f"== {name} ==")
        for m, mk in masks.items():
            print(f"  {m:8s} A={rmse(f_a, y, mk):.3f} B={rmse(f_b, y, mk):.3f} "
                  f"g_mean={g[mk].mean():.3f} g_std={g[mk].std():.3f}")
        # oracle 择优上界
        best = np.where(np.abs(y - f_b) < np.abs(y - f_a), f_b, f_a)
        print(f"  oracleAB overall={rmse(best, y, masks['overall']):.3f} "
              f"sunny={rmse(best, y, sunny):.3f} ramp={rmse(best, y, masks['ramp']):.3f}")
        # 相关性：g 与 |ΔP|（门控是否对爬坡敏感）
        r = np.corrcoef(g, np.abs(dP))[0, 1]
        print(f"  corr(g,|dP|)={r:.3f}")


if __name__ == "__main__":
    main()
