# -*- coding: utf-8 -*-
"""v1.3 天气场景扩展评价（范文 Table 3 的 Sunny/Cloudy/Rainy 三分类）。

数据源：各模型 3 种子的 {stem}_valfull.npz（val 天 = day-block 20%，
训练从未见过的日期；stride=1 全量锚点）。预测取 3 种子集成（均值），
与主表口径一致。
输出：results/WEATHER_EXT.md / WEATHER_EXT.json
      方法 × 尺度(5/15/30) × 天气(晴/阴/雨) × MAE/RMSE/R² + 配对显著性
      （主方法集成 vs 最强 baseline 集成，逐样本平方误差配对 t）。
注：val 曾用于早停（对所有模型同等），解读见 BENCHMARK_LOCK_v1.3_extension.md。
"""
import datetime
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
HORIZONS = [15, 30, 5]
WEATHERS = ["sunny", "cloudy", "rainy"]
METRICS = ["mae", "rmse", "r2"]


def latest_stem(key, h, seed):
    """优先精确名；否则取 _r1/_r2 变体中最旧未被覆盖逻辑影响者（mtime 最新的 ckpt 对应 run）。
    sunset/convlstm h15 为重跑版（_r1），带 valfull。"""
    htag = "" if h == 15 else f"_h{h}"
    base = os.path.join(R, f"{key}{htag}_s{seed}")
    if os.path.exists(base + "_valfull.npz"):
        return base
    k = 1
    cands = []
    while os.path.exists(f"{base}_r{k}_valfull.npz"):
        cands.append(f"{base}_r{k}")
        k += 1
    if not cands:
        return None
    return max(cands, key=lambda p: os.path.getmtime(p + "_valfull.npz"))


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def paired_t(e1, e2):
    d = e1 - e2
    sd = d.std(ddof=1)
    if sd < 1e-12:
        return 0.0, 1.0
    t = d.mean() / (sd / math.sqrt(len(d)))
    return t, 2.0 * (1.0 - norm_cdf(abs(t)))


def main():
    wd = json.load(open(os.path.join(R, "weather_days.json")))
    day_cls = {}
    for w in WEATHERS:
        for d in wd[w]:
            day_cls[d] = w

    cell, data = {}, {}
    for disp, key, _ in METHODS:
        data[key] = {}
        for h in HORIZONS:
            stems = [latest_stem(key, h, s) for s in (0, 1, 2)]
            if any(x is None for x in stems):
                missing = [s for s, x in zip((0, 1, 2), stems) if x is None]
                print(f"SKIP {key}@{h}min: valfull missing seeds {missing}")
                continue
            arrs = [np.load(x + "_valfull.npz") for x in stems]
            # 三种子的锚点必须一致（同 day_block_split + 同 horizon 样本索引）
            at0 = arrs[0]["anchor_time"]
            ok = all((a["anchor_time"] == at0).all() for a in arrs[1:])
            if not ok:
                print(f"WARN {key}@{h}: anchor mismatch across seeds, using seed0 order")
            y = arrs[0]["y"].astype(np.float64)
            ens = np.mean([a["pred"] for a in arrs], axis=0).astype(np.float64)
            dates = np.array([str(np.datetime64(t, "s").astype(datetime.datetime).date())
                              for t in at0])
            masks = {w: np.array([day_cls.get(d, "") == w for d in dates])
                     for w in WEATHERS}
            data[key][h] = {"y": y, "ens": ens, "masks": masks, "n": len(y)}
            cell.setdefault(key, {})[h] = {}
            for w, m in masks.items():
                if m.sum() < 30:
                    cell[key][h][w] = None
                    continue
                e = ens[m] - y[m]
                cell[key][h][w] = {
                    "mae": float(np.mean(np.abs(e))),
                    "rmse": float(np.sqrt(np.mean(e ** 2))),
                    "r2": float(C.r2_score(y[m], ens[m])),
                    "n": int(m.sum())}

    # ---- markdown ----
    lines = ["# 天气场景扩展表（范文 Table 3 复刻：Sunny / Cloudy / Rainy）", "",
             "数据 = val 天（day-block 20%，训练未见日期）stride=1 全量锚点；",
             "主结果 = 3 种子预测集成；单位 kW。粗体 = 该列最优（含 Proposed 参与竞争）。",
             "雨天 = 全天严重遮挡日（晴空指数 <0.45，分类器在官方 20 测试日 20/20 校准）。",
             ""]
    name_map = {k: n for n, k, _ in METHODS}
    order = [k for _, k, _ in METHODS if k in cell]
    for h in HORIZONS:
        rows = [k for k in order if h in cell[k]]
        if not rows:
            continue
        lines += [f"\n## 预测尺度：{h} min\n"]
        hdr = "| Method |"
        sep = "|---|"
        for m in METRICS:
            for w in WEATHERS:
                hdr += f" {m.upper()} {w} |"
                sep += "---:|"
        lines += [hdr, sep]
        for k in rows:
            row = f"| {name_map[k]} |"
            for m in METRICS:
                for w in WEATHERS:
                    c = cell[k][h].get(w)
                    if c is None:
                        row += " - |"
                        continue
                    vals = [(cell[k2][h][w][m], k2) for k2 in rows
                            if cell[k2][h].get(w)]
                    best = (min if m != "r2" else max)(v for v, _ in vals)
                    s = f"{c[m]:.4f}" + (f" (n={c['n']})" if m == "mae" else "")
                    if c[m] == best:
                        s = f"**{s}**"
                    row += f" {s} |"
            lines.append(row)

    # ---- 显著性：主方法 vs 最强 baseline（rainy 与全 val）----
    prop = METHODS[0][1]
    sig = {}
    for h in HORIZONS:
        if h not in cell.get(prop, {}):
            continue
        dp = data[prop][h]
        cand = [k for k in cell if h in cell[k] and k != prop
                and cell[k][h].get("rainy")]
        if not cand:
            continue
        best_bl = min(cand, key=lambda k: cell[k][h]["rainy"]["rmse"])
        allm = np.ones(len(dp["y"]), bool)
        for w, msk in list(dp["masks"].items()) + [("all", allm)]:
            m2 = data[best_bl][h]["masks"][w] if w != "all" else allm
            e1 = (dp["ens"][msk] - dp["y"][msk]) ** 2
            e2 = (data[best_bl][h]["ens"][m2] - dp["y"][msk]) ** 2
            t, p = paired_t(e2, e1)
            sig[f"{w}@{h}"] = {"vs": best_bl, "t": round(t, 2), "p": p,
                               "n": int(msk.sum())}
    lines += ["", "## 显著性（主方法集成 vs 雨 RMSE 最强 baseline，平方误差配对 t）", ""]
    for k, v in sig.items():
        lines.append(f"- {k}: vs {name_map[v['vs']]}  t={v['t']}  p={v['p']:.2e}  n={v['n']}")

    with open(os.path.join(R, "WEATHER_EXT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    json.dump({"cell": {k: {str(h): cell[k][h] for h in cell[k]} for k in cell},
               "sig": sig}, open(os.path.join(R, "WEATHER_EXT.json"), "w"),
              ensure_ascii=False, indent=1)
    print("saved WEATHER_EXT.md / WEATHER_EXT.json")
    for k in cell:
        for h in cell[k]:
            print(name_map[k], h, {w: (round(v["rmse"], 3) if v else None)
                                   for w, v in cell[k][h].items()})


if __name__ == "__main__":
    main()
