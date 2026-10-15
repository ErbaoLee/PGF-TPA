# -*- coding: utf-8 -*-
"""
SKIPP'D benchmark 数据管道与评估（协议 v1.1，自建窗口版）。

数据源：solarbench/SKIPPD（HF，原作者参与的 SolarBench 项目；本地经
Kaggle 镜像获取）——64x64 天空图像 + 同时刻 PV 功率(kW) 的分钟级序列，
train 349372 / test 14003 行（2017–2019，Stanford 30kW 系统）。

forecast 窗口（协议 v1.1）：锚点 t（分钟），输入=[t-14,t] 的 15 帧图像
与 15 个 PV 值，目标 = t+15min 的 PV。仅接受窗口内时间严格连续(60s)且
t+15 存在的锚点。test 重建样本 n=11180。

test 官方晴/阴划分（Nie et al. 2023 固定日期）保留；trainval 的 train/val
按天块 shuffle(seed=0) 80/20 切分（同一天样本绝不跨集，防泄漏）。

指标：RMSE/MAE/R2（sunny/cloudy/overall，kW）+ Forecast Skill
（分母 = RAW persistence，即 pv_log 末值；选择 RAW 为分母的依据：
其 cloudy/overall RMSE 与官方论文 FS 反推值吻合至 0.3%/2%）。
"""
import datetime
import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("PGF_DATA_DIR", os.path.join(ROOT, "data"))
PROC = os.path.join(DATA_DIR, "proc")
RESULTS_DIR = os.environ.get("PGF_RESULTS_DIR", os.path.join(ROOT, "results"))

SUNNY_DATES = [(2017, 9, 15), (2017, 10, 6), (2017, 10, 22),
               (2018, 2, 16), (2018, 6, 12), (2018, 6, 23),
               (2019, 1, 25), (2019, 6, 23), (2019, 7, 14), (2019, 10, 14)]
CLOUDY_DATES = [(2017, 6, 24), (2017, 9, 20), (2017, 10, 11),
                (2018, 1, 25), (2018, 3, 9), (2018, 10, 4),
                (2019, 5, 27), (2019, 6, 28), (2019, 8, 10), (2019, 10, 19)]


def _load_split(split, horizon=15):
    meta = np.load(os.path.join(PROC, f"{split}_meta.npz"))
    # v1.2 扩展：horizon=5 读 _h5 样本索引（目标=t+5min）；默认 15 为
    # v1.1 锁定协议原文件，行为完全不变
    suf = "" if horizon == 15 else f"_h{horizon}"
    smp = np.load(os.path.join(PROC, f"{split}_samples{suf}.npz"))
    images = np.load(os.path.join(PROC, f"{split}_images.dat.npy"), mmap_mode="r")
    return {"images": images, "pv": meta["pv"], "minutes": meta["minutes"],
            "win": smp["win"], "tgt": smp["tgt"], "anchor_time": smp["anchor_time"],
            "horizon": horizon}


_SPLITS = {}


def get_split(split, horizon=15):
    key = (split, horizon)
    if key not in _SPLITS:
        _SPLITS[key] = _load_split(split, horizon)
    return _SPLITS[key]


class ForecastDataset(Dataset):
    """返回 ((15,3,64,64) float 图像帧序列, pv_log(15,), pv_pred 标量)。

    csi=True 时额外返回物理量：x_img, csi_log(15,), p_theo_log(15,),
    p_theo_next(标量), y。csi_log = pv_log/P_theo_log（clip 到 [0,1.5]）。
    防泄漏：sample_idx 必须来自 day_block_split 的输出；本类不做任何划分。
    """

    def __init__(self, split, sample_idx=None, stride=1, no_image=False, csi=False,
                 target_nowcast=False, horizon=15, time_feats=False):
        self.target_nowcast = target_nowcast  # 预训练用：目标=窗口末帧时刻 PV（无未来信息）
        self.split_name = split
        self.horizon = horizon
        self.time_feats = time_feats  # Sky-BiMamba 复现用：12 维多尺度正弦时间编码
        d = get_split(split, horizon)
        self.d = d
        self.no_image = no_image
        self.csi = csi
        if csi:
            cs = np.load(os.path.join(PROC, f"{split}_clearsky.npz"))["p_theo"]
            self.p_theo = cs
        idx = np.arange(len(d["anchor_time"])) if sample_idx is None else np.asarray(sample_idx)
        self.sample_idx = idx[::stride]

    def __getstate__(self):
        st = self.__dict__.copy()
        st["d"] = None  # 4.3GB memmap 不随 pickle 走，worker 内懒重建
        return st

    def __setstate__(self, st):
        self.__dict__.update(st)
        if self.d is None:
            self.d = get_split(self.split_name, self.horizon)

    def __len__(self):
        return len(self.sample_idx)

    def __getitem__(self, i):
        k = int(self.sample_idx[i])
        d = self.d
        rows = d["win"][k]
        pv_log = d["pv"][rows].astype(np.float32)
        # nowcast 预训练目标 = 末帧时刻功率；forecast 目标 = t+horizon min
        y = float(pv_log[-1]) if self.target_nowcast else float(d["pv"][d["tgt"][k]])
        if self.no_image:
            x_img = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
        else:
            arr = np.asarray(d["images"][rows])  # (15,64,64,3) uint8
            x_img = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
        if self.time_feats:
            tf = torch.from_numpy(
                time_encoding_window(d["anchor_time"][k], n=len(rows)))
        if not self.csi:
            if self.time_feats:
                return x_img, torch.from_numpy(pv_log), torch.tensor(y), tf
            return x_img, torch.from_numpy(pv_log), torch.tensor(y)
        pt_log = self.p_theo[rows].astype(np.float32)
        pt_next = float(self.p_theo[d["tgt"][k]])
        csi_log = np.clip(pv_log / np.maximum(pt_log, 0.05), 0.0, 1.5).astype(np.float32)
        if self.time_feats:
            return (x_img, torch.from_numpy(pv_log), torch.from_numpy(csi_log),
                    torch.from_numpy(pt_log), torch.tensor(pt_next), torch.tensor(y), tf)
        return (x_img, torch.from_numpy(pv_log), torch.from_numpy(csi_log),
                torch.from_numpy(pt_log), torch.tensor(pt_next), torch.tensor(y))


def day_block_split(anchor_time, seed=0, val_frac=0.2):
    """按锚点所在日期做天块 shuffle 后 80/20 切分。返回 (train_idx, val_idx)。"""
    rng = np.random.RandomState(seed)
    dates = _anchor_dates(anchor_time)
    uniq = np.unique(dates)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    val_days = set(uniq[:n_val])
    val_mask = np.array([d in val_days for d in dates])
    return np.where(~val_mask)[0], np.where(val_mask)[0]


def day_block_kfold(anchor_time, n_folds=5, seed=0):
    """官方同款 day-block K 折：天 shuffle 后均分 n_folds 组，
    第 k 组整组作 val。yield (train_idx, val_idx, fold_id)。"""
    rng = np.random.RandomState(seed)
    dates = _anchor_dates(anchor_time)
    uniq = np.unique(dates)
    rng.shuffle(uniq)
    folds = np.array_split(uniq, n_folds)
    for k in range(n_folds):
        val_days = set(folds[k])
        val_mask = np.array([d in val_days for d in dates])
        yield np.where(~val_mask)[0], np.where(val_mask)[0], k


def _anchor_dates(anchor_time):
    return np.array([(np.datetime64(t, "s")).astype(datetime.datetime).date()
                     for t in anchor_time])


def time_encoding(anchor_sec):
    """Sky-BiMamba（范文 §2.3，Eq.10-13）12 维多尺度正弦时间编码。
    输入时刻（np.datetime64[s]），输出 float32(12,)：
    E_base(4)=小时/分钟 sin-cos；E_mid(6)=年内日序 7/14/21 天周期；
    E_long(2)=月份 sin-cos。"""
    t = (np.datetime64(anchor_sec, "s")).astype(datetime.datetime)
    ho, mi, da, mo = t.hour, t.minute, t.timetuple().tm_yday, t.month
    e_base = [np.sin(2 * np.pi * ho / 24), np.cos(2 * np.pi * ho / 24),
              np.sin(2 * np.pi * mi / 60), np.cos(2 * np.pi * mi / 60)]
    e_mid = []
    for p in (7, 14, 21):
        e_mid += [np.sin(2 * np.pi * da / p), np.cos(2 * np.pi * da / p)]
    e_long = [np.sin(2 * np.pi * (mo - 1) / 12), np.cos(2 * np.pi * (mo - 1) / 12)]
    return np.array(e_base + e_mid + e_long, dtype=np.float32)


def time_encoding_window(anchor_sec, n=15):
    """窗口内 n 个时刻（[t−n+1, t]，严格 60s 连续）的逐时刻编码，(n,12)。"""
    a = np.datetime64(anchor_sec, "s")
    return np.stack([time_encoding(a - np.timedelta64((n - 1 - i) * 60, "s"))
                     for i in range(n)])


def sunny_cloudy_masks(anchor_time):
    dates = np.array([(np.datetime64(t, "s")).astype(datetime.datetime).date()
                      for t in anchor_time])
    sunny = np.zeros(len(dates), dtype=bool)
    for dd in SUNNY_DATES:
        sunny[np.where(dates == datetime.date(*dd))[0]] = True
    cloudy = np.zeros(len(dates), dtype=bool)
    for dd in CLOUDY_DATES:
        cloudy[np.where(dates == datetime.date(*dd))[0]] = True
    return sunny, cloudy


def r2_score(y, yhat):
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - np.sum((y - yhat) ** 2) / ss_tot if ss_tot > 0 else float("nan")


def evaluate(y_true, y_pred, anchor_time, tag="", unit="kW"):
    """官方口径：sunny / cloudy / overall 的 RMSE、MAE、R2。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    sunny, cloudy = sunny_cloudy_masks(anchor_time)
    out = {"unit": unit}
    for name, m in [("sunny", sunny), ("cloudy", cloudy),
                    ("overall", np.ones(len(y_true), dtype=bool))]:
        if not m.any():
            out[name] = {"rmse": None, "mae": None, "r2": None, "n": 0}
            continue
        out[name] = {"rmse": float(np.sqrt(np.mean((y_pred[m] - y_true[m]) ** 2))),
                     "mae": float(np.mean(np.abs(y_pred[m] - y_true[m]))),
                     "r2": float(r2_score(y_true[m], y_pred[m])),
                     "n": int(m.sum())}
    if tag:
        out["tag"] = tag
    return out


def forecast_skill(rmse_model, rmse_ref):
    return 1.0 - rmse_model / rmse_ref


def csi_persistence(anchor_time, pv_log, y_true, horizon=15):
    """官方 clear-sky-index persistence：pred = CSI(t)*P_clearsky(t+H)。
    P_clearsky 由官方 Relative_op_func 解析计算（kW）。结果缓存。"""
    cache = os.path.join(RESULTS_DIR,
                         "csi_persistence_cache.npz" if horizon == 15
                         else f"csi_persistence_cache_h{horizon}.npz")
    key = hash((len(anchor_time), str(anchor_time[0]), str(anchor_time[-1])))
    if os.path.exists(cache):
        c = np.load(cache, allow_pickle=True)
        if int(c["key"]) == key:
            return c["preds"]
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rel_op", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "relative_op_func_reference.py"))
    rel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rel)
    preds = np.zeros(len(anchor_time))
    for i in range(len(anchor_time)):
        t = (np.datetime64(anchor_time[i], "s")).astype(datetime.datetime)
        try:
            csi_now, _ = rel.Relative_output(t, float(pv_log[i][0]))
            _, p_next = rel.Relative_output(t + datetime.timedelta(minutes=horizon),
                                            float(y_true[i]))
            preds[i] = csi_now * p_next
        except Exception:
            preds[i] = pv_log[i][-1]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(cache, key=key, preds=preds)
    return preds


def dump_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print("saved:", path)
