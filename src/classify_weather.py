# -*- coding: utf-8 -*-
"""v1.3 天气场景扩展：全数据日级天气自动分类（晴/阴/雨）。

设计（对齐范文 Table 3 的 Sunny/Cloudy/Rainy 三分类）：
  - 范文自建 16 个测试日（4晴/4阴/4雨），人工按天空图像分类；官方 SKIPP'D
    test 只有 10 晴 + 10 阴，无雨天。
  - 本脚本用**可复现的规则分类器**代替人工：日级特征 = 晴空指数
    (白天 PV 能量 / 晴空理论能量) + 归一化 1-min 波动率 + 天空图像
    亮度/饱和度统计（雨天=深灰低饱和）。
  - **阈值校准只用官方 20 个测试日的公开晴/阴标签**（Nie et al. 2023
    固定日期清单），要求 20/20 全对，再外推到全部 497 个训练日。
  - 雨天场景从 val 天（day-block 20%，训练从未见过）中选取，避免任何
    训练泄漏；早停曾用 val 的注意见 BENCHMARK_LOCK_v1.3_extension.md。
输出：
  data/proc/day_weather.json   全部日期特征+分类
  results/weather_days.json    val 天三分组（晴/阴/雨日期清单）
  results/weather_montage_*.png 雨天候选蒙太奇（人工核对用）
"""
import datetime
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

PROC = C.PROC
R = C.RESULTS_DIR


def day_features(split):
    meta = np.load(os.path.join(PROC, f"{split}_meta.npz"))
    cs = np.load(os.path.join(PROC, f"{split}_clearsky.npz"))["p_theo"]
    minutes = meta["minutes"].astype("datetime64[s]")
    pv = meta["pv"].astype(np.float64)
    frame_rows = meta["frame_rows"]
    images = np.load(os.path.join(PROC, f"{split}_images.dat.npy"), mmap_mode="r")

    dates = np.array([(np.datetime64(t, "s")).astype(datetime.datetime).date()
                      for t in minutes])
    feats = {}
    for d in np.unique(dates):
        m = dates == d
        pt = cs[m]
        pv_d = pv[m]
        day = pt > 0.5  # 白天（晴空理论功率足够大）
        if day.sum() < 60:  # <1h 白天数据，跳过
            continue
        e_pv, e_theo = pv_d[day].sum(), pt[day].sum()
        clearness = float(e_pv / max(e_theo, 1.0))
        # 归一化 1-min 波动率（晴天阵风/云遮挡的瞬时抖动）
        ci = pv_d[day] / np.maximum(pt[day], 0.5)  # 逐分钟晴空指数
        var_n = float(np.mean(np.abs(np.diff(ci))))
        rri = float(ci.max() - ci.min())  # 日内晴空指数极差（云遮挡幅度）
        # 图像统计：取白天中间 6 帧的亮度/饱和度（uint8 RGB memmap）
        glob_idx = np.where(m)[0]                  # 全局行号
        bright_idx = glob_idx[pt > 5.0]            # 晴空理论 >5kW 的白天帧
        if len(bright_idx) >= 6:
            pick = bright_idx[np.linspace(0, len(bright_idx) - 1, 6).astype(int)]
            lum, sat = [], []
            for r_ in pick:
                fr = frame_rows[r_]
                if fr < 0 or fr >= len(images):
                    continue
                img = np.asarray(images[fr]).astype(np.float32) / 255.0
                lum.append(float(img.mean()))
                mx, mn = img.max(axis=2), img.min(axis=2)
                sat.append(float(np.mean((mx - mn) / np.maximum(mx, 1e-6))))
            bright, sat = float(np.mean(lum)), float(np.mean(sat))
        else:
            bright, sat = float("nan"), float("nan")
        feats[str(d)] = {"clearness": round(clearness, 4),
                         "var_n": round(var_n, 4), "rri": round(rri, 4),
                         "bright": None if np.isnan(bright) else round(bright, 4),
                         "sat": None if np.isnan(sat) else round(sat, 4),
                         "split": split}
    return feats


def classify(f):
    """规则分类（阈值在官方 20 测试日上校准 20/20，见 main 的校准报告）：
    晴/阴由波动率 var_n 完美分离（官方晴日 ≤0.0126 < 0.02 < 官方阴日 ≥0.028）；
    雨天 = 全天严重遮挡（clearness<0.45，官方最阴日 0.471 仍为阴）。"""
    if f["var_n"] < 0.02 and f["clearness"] >= 0.70:
        return "sunny"
    if f["clearness"] < 0.45:
        return "rainy"
    return "cloudy"


def main():
    feats = {}
    feats.update(day_features("test"))
    feats.update(day_features("train"))

    # ---- 校准：官方 20 个测试日必须 20/20 复现 ----
    ok, bad = 0, []
    for dd in C.SUNNY_DATES:
        k = str(datetime.date(*dd))
        pred = classify(feats[k]) if k in feats else "missing"
        if pred == "sunny":
            ok += 1
        else:
            bad.append((k, "sunny", pred))
    for dd in C.CLOUDY_DATES:
        k = str(datetime.date(*dd))
        pred = classify(feats[k]) if k in feats else "missing"
        if pred == "cloudy":
            ok += 1
        else:
            bad.append((k, "cloudy", pred))
    print(f"官方 20 测试日校准：{ok}/20 正确")
    for b in bad:
        print("  MISS", b, feats.get(b[0]))

    for k, f in feats.items():
        f["cls"] = classify(f)

    # val 天重建（与训练协议同规则 seed=0；按日期，与 horizon 无关）
    tr = C.get_split("train")
    tr_idx, va_idx = C.day_block_split(tr["anchor_time"], seed=0, val_frac=0.2)
    va_dates = {str(np.datetime64(t, "s").astype(datetime.datetime).date())
                for t in tr["anchor_time"][va_idx]}

    from collections import Counter
    val_cls = Counter(feats[d]["cls"] for d in va_dates if d in feats)
    all_cls = Counter(f["cls"] for f in feats.values())
    print("全数据分类分布:", dict(all_cls))
    print("val 天分类分布:", dict(val_cls))

    rainy_val = sorted(d for d in va_dates if d in feats and feats[d]["cls"] == "rainy")
    sunny_val = sorted(d for d in va_dates if d in feats and feats[d]["cls"] == "sunny")
    cloudy_val = sorted(d for d in va_dates if d in feats and feats[d]["cls"] == "cloudy")
    print(f"val 雨天 {len(rainy_val)} 天:", rainy_val)
    print(f"val 晴天 {len(sunny_val)} 天，val 阴天 {len(cloudy_val)} 天")

    os.makedirs(R, exist_ok=True)
    with open(os.path.join(PROC, "day_weather.json"), "w") as fp:
        json.dump(feats, fp, indent=1)
    with open(os.path.join(R, "weather_days.json"), "w") as fp:
        json.dump({"rainy": rainy_val, "sunny": sunny_val, "cloudy": cloudy_val,
                   "n_val_days": len(va_dates),
                   "calibration": f"{ok}/20"}, fp, indent=1)
    print("saved day_weather.json / weather_days.json")

    # ---- 雨天候选蒙太奇（人工核对）----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        meta = np.load(os.path.join(PROC, "train_meta.npz"))
        minutes = meta["minutes"].astype("datetime64[s]")
        frame_rows = meta["frame_rows"]
        dates = np.array([str(np.datetime64(t, "s").astype(datetime.datetime).date())
                          for t in minutes])
        images = np.load(os.path.join(PROC, "train_images.dat.npy"), mmap_mode="r")
        for d in rainy_val[:12]:
            m = np.where(dates == d)[0]
            if len(m) < 10:
                continue
            pick = m[np.linspace(0, len(m) - 1, 6).astype(int)]
            fig, axes = plt.subplots(1, 6, figsize=(18, 3))
            for ax, r_ in zip(axes, pick):
                fr = frame_rows[r_]
                if 0 <= fr < len(images):
                    ax.imshow(np.asarray(images[fr]))
                ax.set_xticks([]), ax.set_yticks([])
            f = feats[d]
            fig.suptitle(f"{d}  cl={f['clearness']:.2f} var={f['var_n']:.3f} "
                         f"rri={f['rri']:.2f} bright={f['bright']} sat={f['sat']}")
            fig.tight_layout()
            fig.savefig(os.path.join(R, f"weather_montage_{d}.png"), dpi=72)
            plt.close(fig)
        print("montages saved")
    except Exception as ex:  # 蒙太奇失败不影响主流程
        print("montage skipped:", ex)


if __name__ == "__main__":
    main()
