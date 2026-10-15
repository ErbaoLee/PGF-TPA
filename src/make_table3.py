# -*- coding: utf-8 -*-
"""范文式主表生成器 v2（对齐 Renewable Energy 256 (2026) 124361 表 2/表 3）。

表 2（TABLE2.md）：Framework × {MAE,RMSE,R²} × {5,15,30 min}（overall，官方 test）
表 3（TABLE3.md）：方法 × 天气 × 尺度（sunny/cloudy/overall；rainy 见
        WEATHER_EXT.md——官方 test 无雨天，雨天用 val 天扩展口径）
数据源：results/preds_*.npy（3 seeds 集成为主结果，单 seed 均值±std 附表）。
文件选择：精确名优先，否则取 mtime 最新的 _r{k} 变体（sunset/convlstm h15 重跑版）。
输出：results/TABLE2.md / TABLE3.md / TABLE3.csv / TABLE3_std.md / TABLE3_sig.json
"""
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

R = C.RESULTS_DIR

METHODS = [
    ("PGF-TPA (Proposed)", "pgf_nocsi_tpa", True),
    ("Sky-BiMamba (paper)", "skybimamba", False),
    ("SUNSET", "sunset", False),
    ("ResLSTM", "reslstm", False),
    ("ConvLSTM", "convlstm", False),
    ("Transformer", "transformer", False),
    ("PatchTST", "patchtst", False),
    ("DLinear", "dlinear", False),
    ("CrossGNN", "crossgnn", False),
]
WEATHERS = ["sunny", "cloudy", "overall"]
HORIZONS = [5, 15, 30]
METRICS = ["mae", "rmse", "r2"]


def latest_preds(key, h, seed):
    """返回该 (方法,尺度,种子) 的 preds npy 路径：精确名优先，否则最新 _r{k}。"""
    htag = "" if h == 15 else f"_h{h}"
    base = os.path.join(R, f"preds_{key}{htag}_s{seed}")
    if os.path.exists(base + ".npy"):
        return base + ".npy"
    cands = glob.glob(base + "_r*.npy")
    return max(cands, key=os.path.getmtime) if cands else None


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def paired_t(e1, e2):
    d = e1 - e2
    sd = d.std(ddof=1)
    if sd < 1e-12:
        return 0.0, 1.0
    t = d.mean() / (sd / math.sqrt(len(d)))
    return t, 2.0 * (1.0 - norm_cdf(abs(t)))


def load_all():
    out = {}
    for _, key, _ in METHODS:
        out[key] = {}
        for h in HORIZONS:
            ps = [latest_preds(key, h, s) for s in (0, 1, 2)]
            if any(p is None for p in ps):
                missing = [s for s, p in zip((0, 1, 2), ps) if p is None]
                print(f"SKIP {key}@{h}: missing seeds {missing}")
                continue
            te = C.get_split("test", h)
            y = te["pv"][te["tgt"]].astype(np.float64)
            at = te["anchor_time"]
            seeds = [np.load(p).astype(np.float64) for p in ps]
            for i, s in enumerate(seeds):
                assert len(s) == len(y), f"{key}@{h} seed{i} 长度不符 {len(s)}!={len(y)}"
            ens = np.mean(seeds, axis=0)
            out[key][h] = {"ens": ens, "seeds": seeds, "y": y, "at": at}
    return out


def raw_persistence(h):
    te = C.get_split("test", h)
    return te["pv"][te["win"]][:, -1].astype(np.float64)


def main():
    data = load_all()
    if not data:
        print("no preds found")
        return

    cell, cell_std = {}, {}
    for _, key, _ in METHODS:
        if key not in data:
            continue
        cell[key], cell_std[key] = {}, {}
        for h in HORIZONS:
            if h not in data[key]:
                continue
            d = data[key][h]
            masks = {"sunny": None, "cloudy": None, "overall": np.ones(len(d["y"]), bool)}
            sunny, cloudy = C.sunny_cloudy_masks(d["at"])
            masks["sunny"], masks["cloudy"] = sunny, cloudy
            cell[key][h], cell_std[key][h] = {}, {}
            r = C.evaluate(d["y"], d["ens"], d["at"])
            for w in WEATHERS:
                cell[key][h][w] = {m: r[w][m] for m in METRICS}
            for w, msk in masks.items():
                for m in METRICS:
                    vals = []
                    for s in d["seeds"]:
                        r1 = C.evaluate(d["y"][msk], s[msk], d["at"][msk])
                        vals.append(r1["overall"][m])
                    cell_std[key][h][w] = cell_std[key][h].get(w, {})
                    cell_std[key][h][w][m] = (float(np.mean(vals)), float(np.std(vals)))
            np.save(os.path.join(R, f"preds_TABLE3_{key}_{h}.npy"), d["ens"])

    def is_best(key, h, w, m):
        vals = [(cell[k][h][w][m], k) for k in cell if h in cell[k]
                and w in cell[k][h]]
        if not vals:
            return False
        best_v = (min if m != "r2" else max)(v for v, _ in vals)
        return cell[key][h][w][m] == best_v

    name_map = {k: n for n, k, _ in METHODS}
    order = [k for _, k, _ in METHODS if k in cell]

    # ---------- 表 2（overall × 尺度） ----------
    lines = ["# Table 2（范文式）不同预测尺度下的框架对比（官方 test，overall）", "",
             "主结果 = 3-seed 集成（预测均值）；单位 kW；R² 无量纲。粗体 = 该列最优。",
             "协议：SKIPP'D 官方 test 20 日（10 晴/10 阴），输入 [t−14,t] 15 帧 + PV。",
             ""]
    hdr = "| Framework |"
    sep = "|---|"
    for m in METRICS:
        for h in HORIZONS:
            hdr += f" {m.upper()} {h}min |"
            sep += "---:|"
    lines += [hdr, sep]
    for k in order:
        row = f"| {name_map[k]} |"
        for m in METRICS:
            for h in HORIZONS:
                if h not in cell[k]:
                    row += " - |"
                    continue
                v = cell[k][h]["overall"][m]
                s = f"{v:.4f}"
                if is_best(k, h, "overall", m):
                    s = f"**{s}**"
                row += f" {s} |"
        lines.append(row)
    with open(os.path.join(R, "TABLE2.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("saved TABLE2.md")

    # ---------- 表 3（天气 × 尺度） ----------
    lines = ["# Table 3（范文式）不同天气条件与预测尺度下的框架对比（官方 test）", "",
             "主结果 = 3-seed 集成；单位 kW。粗体 = 该列最优。",
             "官方 test 为 10 晴 + 10 阴（Nie et al. 2023 固定日期，无雨天）→",
             "Rainy 场景见 WEATHER_EXT.md（val 天扩展口径，晴/阴/雨三分类）。",
             ""]
    hdr = "| Method |"
    sep = "|---|"
    for m in METRICS:
        for w in WEATHERS:
            hdr += f" {m.upper()} {w} |"
            sep += "---:|"
    for h in HORIZONS:
        sub = [f"\n## 预测尺度：{h} min\n", hdr, sep]
        for k in order:
            if h not in cell[k]:
                continue
            row = f"| {name_map[k]} |"
            for m in METRICS:
                for w in WEATHERS:
                    if w not in cell[k][h]:
                        row += " - |"
                        continue
                    v = cell[k][h][w][m]
                    s = f"{v:.4f}"
                    if is_best(k, h, w, m):
                        s = f"**{s}**"
                    row += f" {s} |"
            sub.append(row)
        lines += sub

    # FS + persistence 参照
    lines += ["\n## Forecast Skill（FS = 1 − RMSE_model / RMSE_RAW-persistence，同尺度）", ""]
    fs_tbl = {}
    for h in HORIZONS:
        raw = raw_persistence(h)
        te = C.get_split("test", h)
        y = te["pv"][te["tgt"]].astype(np.float64)
        raw_rmse = float(np.sqrt(np.mean((raw - y) ** 2)))
        lines.append(f"- {h} min：RAW persistence overall RMSE = {raw_rmse:.4f} kW")
        for k in order:
            if h in cell[k]:
                fs = 1 - cell[k][h]["overall"]["rmse"] / raw_rmse
                fs_tbl[f"{k}@{h}"] = fs
                lines.append(f"  - {name_map[k]}: FS = {fs:+.1%}")
    lines += ["", "## Persistence 参照（不入主表竞争）", ""]
    for h in HORIZONS:
        te = C.get_split("test", h)
        y = te["pv"][te["tgt"]].astype(np.float64)
        at = te["anchor_time"]
        raw = raw_persistence(h)
        r = C.evaluate(y, raw, at)
        lines.append(f"- RAW persistence @{h}min: overall RMSE {r['overall']['rmse']:.4f} / "
                     f"MAE {r['overall']['mae']:.4f} / R² {r['overall']['r2']:.4f} "
                     f"(sunny {r['sunny']['rmse']:.4f}, cloudy {r['cloudy']['rmse']:.4f})")
        try:
            csi = C.csi_persistence(at, te["pv"][te["win"]].astype(np.float64), y, horizon=h)
            r = C.evaluate(y, csi, at)
            lines.append(f"- CSI persistence @{h}min: overall RMSE {r['overall']['rmse']:.4f} / "
                         f"MAE {r['overall']['mae']:.4f} / R² {r['overall']['r2']:.4f} "
                         f"(sunny {r['sunny']['rmse']:.4f}, cloudy {r['cloudy']['rmse']:.4f})")
        except Exception as ex:
            lines.append(f"- CSI persistence @{h}min: 计算失败 {ex}")

    # 显著性
    prop = METHODS[0][1]
    sig = {}
    for h in HORIZONS:
        if h not in cell.get(prop, {}):
            continue
        dp = data[prop][h]
        best_bl, best_r = None, np.inf
        for k in order:
            if k == prop or h not in cell[k]:
                continue
            if cell[k][h]["overall"]["rmse"] < best_r:
                best_r, best_bl = cell[k][h]["overall"]["rmse"], k
        if best_bl is None:
            continue
        sunny, cloudy = C.sunny_cloudy_masks(dp["at"])
        for w, msk in [("sunny", sunny), ("cloudy", cloudy),
                       ("overall", np.ones(len(dp["y"]), bool))]:
            e_prop = (dp["ens"][msk] - dp["y"][msk]) ** 2
            e_bl = (data[best_bl][h]["ens"][msk] - dp["y"][msk]) ** 2
            t, p = paired_t(e_bl, e_prop)
            sig[f"{w}@{h}"] = {"vs": best_bl, "t": round(t, 2), "p": p,
                               "n": int(msk.sum())}
    lines += ["", "## 显著性（主方法集成 vs 最强 baseline 集成，平方误差配对 t，双侧）", ""]
    for k, v in sig.items():
        lines.append(f"- {k}: vs {name_map[v['vs']]}  t={v['t']}  p={v['p']:.2e}  n={v['n']}")

    with open(os.path.join(R, "TABLE3.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("saved TABLE3.md")

    rows = ["method,horizon,weather,mae,rmse,r2"]
    for k in order:
        for h in HORIZONS:
            if h not in cell[k]:
                continue
            for w in WEATHERS:
                c = cell[k][h][w]
                rows.append(f"{name_map[k]},{h},{w},{c['mae']:.6f},{c['rmse']:.6f},{c['r2']:.6f}")
    with open(os.path.join(R, "TABLE3.csv"), "w") as f:
        f.write("\n".join(rows) + "\n")
    print("saved TABLE3.csv")

    lines = ["# TABLE3 附表：单模型 3-seed 均值 ± 标准差（集成见主表）", ""]
    for h in HORIZONS:
        lines += [f"\n## {h} min\n", hdr, sep]
        for k in order:
            if h not in cell_std[k]:
                continue
            row = f"| {name_map[k]} |"
            for m in METRICS:
                for w in WEATHERS:
                    mu, sd = cell_std[k][h][w][m]
                    row += f" {mu:.4f}±{sd:.4f} |"
            lines.append(row)
    with open(os.path.join(R, "TABLE3_std.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("saved TABLE3_std.md")

    json.dump({"cell": {k: {str(h): cell[k][h] for h in cell[k]} for k in cell},
               "significance": sig, "fs": fs_tbl},
              open(os.path.join(R, "TABLE3_sig.json"), "w"), ensure_ascii=False, indent=1)
    print("saved TABLE3_sig.json")


if __name__ == "__main__":
    main()
