# -*- coding: utf-8 -*-
"""编码器失败模式诊断（推理级，不训练）——为"自创编码器替代 ConvLSTM"提供
failure-driven 依据。

对照模型（同 PGF 框架、同协议、同 ckpt 复现数字）：
  - CLB：B 分支 = 共享空间卷积 + ConvLSTM 递归（E37 成员同款）
  - CNN：B 分支 = 15 帧拼通道静态卷积（v2.1）

诊断项（test 集，帧替换 = 该窗口 15 帧逐像素均值图，保持亮度抹掉云信息）：
  occ_first5 / occ_mid5 / occ_last5 : 遮挡早/中/晚 5 帧（信息依赖的时间分布）
  reverse : 帧序反转（时序方向敏感性）
  shuffle : 帧序随机置换（时序结构依赖）

解读规则（预注册）：
  - 若 CLB 遮早期帧的损伤显著小于 CNN → 递归链丢弃早期帧信息（15 步衰减）
    → 新编码器需并行保留全时程信息；
  - 若两者对最近帧同样敏感 → recency 是任务属性，不是 ConvLSTM 缺陷；
  - reverse/shuffle 损伤小 → 编码器实际用时序动力学少（学的是"当前天空状态"）。
"""
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from models import PGF  # noqa: E402

RESULTS = os.path.join(C.RESULTS_DIR, "encoder_diag.json")


@torch.no_grad()
def run_variant(model, dl, dev, variant, rng_seed=0):
    model.eval()
    preds, ys = [], []
    for x_img, pv_log, csi_log, pt_log, pt_next, y in dl:
        x = x_img.to(dev)
        if variant != "clean":
            mean_frame = x.mean(dim=1, keepdim=True)  # (B,1,3,64,64) 窗口均值帧
            if variant.startswith("occ"):
                lo, hi = {"first5": (0, 5), "mid5": (5, 10), "last5": (10, 15)}[
                    variant.split("_")[1]]
                x = x.clone()
                x[:, lo:hi] = mean_frame.expand(-1, hi - lo, -1, -1, -1)
            elif variant == "reverse":
                x = torch.flip(x, dims=[1])
            elif variant == "shuffle":
                g = torch.Generator().manual_seed(rng_seed)
                perm = torch.randperm(x.shape[1], generator=g).to(x.device)
                x = x[:, perm]
        preds.append(model(x, pv_log.to(dev), csi_log.to(dev),
                           pt_next.to(dev)).float().cpu().numpy())
        ys.append(y.numpy())
    p, y = np.concatenate(preds), np.concatenate(ys)
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    return rmse, p, y


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    te = C.ForecastDataset("test", None, stride=1, csi=True)
    dl = DataLoader(te, batch_size=512, shuffle=False, num_workers=0)
    at = C.get_split("test")["anchor_time"]
    sunny, cloudy = C.sunny_cloudy_masks(at)
    y_true = C.get_split("test")["pv"][C.get_split("test")["tgt"]].astype(np.float64)

    ckpts = {
        "clb_s0": ("results/pgf_nocsi_clb_s0_ckpt.pt", dict(use_cl_b=True)),
        "cnn_f0": ("results/pgf_nocsi_s1_f0_ckpt.pt", dict(use_cl_b=False)),
    }
    variants = ["clean", "occ_first5", "occ_mid5", "occ_last5", "reverse", "shuffle"]
    out = {}
    for name, (path, kw) in ckpts.items():
        model = PGF(use_csi=False, **kw).to(dev)
        sd = torch.load(os.path.join(C.ROOT, path), map_location=dev)
        model.load_state_dict(sd)
        model.eval()
        rows = {}
        for v in variants:
            rmse, p, y = run_variant(model, dl, dev, v)
            rows[v] = {"overall": rmse,
                       "sunny": float(np.sqrt(np.mean((p[sunny] - y[sunny]) ** 2))),
                       "cloudy": float(np.sqrt(np.mean((p[cloudy] - y[cloudy]) ** 2)))}
            print(f"{name:8s} {v:12s} overall={rmse:.4f} "
                  f"sunny={rows[v]['sunny']:.4f} cloudy={rows[v]['cloudy']:.4f}",
                  flush=True)
        base = rows["clean"]["overall"]
        rows["delta_vs_clean"] = {v: round(rows[v]["overall"] - base, 4)
                                  for v in variants if v != "clean"}
        out[name] = rows
    import json
    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("saved:", RESULTS)


if __name__ == "__main__":
    main()
