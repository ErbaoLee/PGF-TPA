# -*- coding: utf-8 -*-
"""阶段四评估：AWA 编码器 vs ConvLSTM(B=CLB)同协议对比 + 机制消融。

对比矩阵（全部逐样本平方误差配对 t，负值=左项更优）：
  单模型逐 seed： pgf_nocsi_awa_s{k} vs pgf_nocsi_clb_s{k}
  3-seed 集成：   AWA-ens vs CLB-ens
  消融(s0)：      awa vs awa-noalign(位移置零) / awa vs awa-noextrap(无外推)
  外部锚点：      AWA-ens vs ConvLSTM_ens / SUNSET_ens / E19 / E37
分块：overall / sunny / cloudy / ramp10(|ΔP|>3kW) / non-ramp。
若存在 fold 版成员(preds_pgf_nocsi_awa_s*_f*.npy)自动扩为全网格集成对比。
输出 results/awa_summary.json。
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def L(n):
    p = os.path.join(C.RESULTS_DIR, f"preds_{n}.npy")
    if not os.path.exists(p):
        return None
    return np.load(p).astype(np.float64)


def paired_t(p1, p2, y, m=None):
    m = np.ones(len(y), bool) if m is None else m
    d = (p1[m] - y[m]) ** 2 - (p2[m] - y[m]) ** 2
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))


def main():
    te = C.get_split("test")
    at = te["anchor_time"]
    y = te["pv"][te["tgt"]].astype(np.float64)
    p0 = te["pv"][te["win"]][:, -1].astype(np.float64)
    dP = y - p0
    sunny, cloudy = C.sunny_cloudy_masks(at)
    masks = {"overall": np.ones(len(y), bool), "sunny": sunny, "cloudy": cloudy,
             "ramp10": np.abs(dP) > 3.0, "nonramp": np.abs(dP) <= 3.0,
             "up10": dP > 3.0, "down10": dP < -3.0}

    def stats(p):
        return {k: round(float(np.sqrt(np.mean((p[m] - y[m]) ** 2))), 4)
                for k, m in masks.items()}

    out = {"n_test": len(y)}

    # 单模型
    singles = {}
    for k in range(3):
        a, b = L(f"pgf_nocsi_awa_s{k}"), L(f"pgf_nocsi_clb_s{k}")
        if a is not None and b is not None:
            singles[f"awa_s{k}"] = stats(a)
            singles[f"clb_s{k}"] = stats(b)
            singles[f"t_awa_vs_clb_s{k}"] = {mm: round(paired_t(a, b, y, m), 2)
                                             for mm, m in masks.items()}
    out["singles"] = singles

    # TPA（终版：帧共享深spatial+时间池化，无递归无warp）
    tpa3 = [L(f"pgf_nocsi_tpa_s{k}") for k in range(3)]
    if all(x is not None for x in tpa3):
        tp = {}
        for k in range(3):
            tp[f"tpa_s{k}"] = stats(tpa3[k])
            b = L(f"pgf_nocsi_clb_s{k}")
            if b is not None:
                tp[f"t_tpa_vs_clb_s{k}"] = {mm: round(paired_t(tpa3[k], b, y, m), 2)
                                            for mm, m in masks.items()}
        clb3_ = [x for x in [L(f"pgf_nocsi_clb_s{k}") for k in range(3)]
                 if x is not None]
        if len(clb3_) == 3:
            A, B = np.mean(tpa3, axis=0), np.mean(clb3_, axis=0)
            np.save(os.path.join(C.RESULTS_DIR, "preds_tpa_ens3.npy"), A)
            tp["tpa_ens3"] = stats(A)
            tp["clb_ens3"] = stats(B)
            tp["t_tpaens_vs_clbens"] = {mm: round(paired_t(A, B, y, m), 2)
                                        for mm, m in masks.items()}
            for nm, r in [("ConvLSTM_ens", L("convlstm_ensemble")),
                          ("E19", L("pgf_E19_all")), ("E37", L("pgf_E37"))]:
                if r is not None:
                    tp[f"t_tpaens_vs_{nm}"] = {mm: round(paired_t(A, r, y, m), 2)
                                               for mm, m in masks.items()}
        out["tpa"] = tp

    # 3-seed 集成
    awa3 = [L(f"pgf_nocsi_awa_s{k}") for k in range(3)]
    clb3 = [L(f"pgf_nocsi_clb_s{k}") for k in range(3)]
    if all(x is not None for x in awa3) and all(x is not None for x in clb3):
        A, B = np.mean(awa3, axis=0), np.mean(clb3, axis=0)
        np.save(os.path.join(C.RESULTS_DIR, "preds_awa_ens3.npy"), A)
        out["ens3"] = {"awa": stats(A), "clb": stats(B),
                       "t_awa_vs_clb": {mm: round(paired_t(A, B, y, m), 2)
                                        for mm, m in masks.items()}}

    # 消融（同 seed s0）
    for abl, nm in [("pgf_nocsi_awa_awana_s0", "noalign"),
                    ("pgf_nocsi_awa_awane_s0", "noextrap")]:
        p = L(abl)
        base = L("pgf_nocsi_awa_s0")
        if p is not None and base is not None:
            out[f"abl_{nm}"] = {nm: stats(p), "awa": stats(base),
                                f"t_awa_vs_{nm}": {mm: round(paired_t(base, p, y, m), 2)
                                                   for mm, m in masks.items()}}

    # 全网格集成（若 fold 存在）
    folds = sorted(glob.glob(os.path.join(C.RESULTS_DIR, "preds_pgf_nocsi_awa_s*_f*.npy")))
    if folds:
        members = [np.load(f).astype(np.float64) for f in folds]
        members += [x for x in awa3 if x is not None]
        E = np.mean(members, axis=0)
        np.save(os.path.join(C.RESULTS_DIR, "preds_awa_fullens.npy"), E)
        row = {"n_members": len(members), "awa_full": stats(E)}
        for ref, rn in [("preds_pgf_E37", "E37"), ("preds_pgf_E18_clb", "E18_clb"),
                        ("preds_awa_ens3", "awa_ens3")]:
            r = L(ref.replace("preds_", ""))
            if r is not None:
                row[f"t_vs_{rn}"] = {mm: round(paired_t(E, r, y, m), 2)
                                     for mm, m in masks.items()}
                row[rn] = stats(r)
        out["fullens"] = row

    # TPA 全网格集成（若 fold 存在）
    tfolds = sorted(glob.glob(os.path.join(C.RESULTS_DIR, "preds_pgf_nocsi_tpa_s*_f*.npy")))
    if tfolds and all(x is not None for x in tpa3):
        members = [np.load(f).astype(np.float64) for f in tfolds] + tpa3
        E = np.mean(members, axis=0)
        np.save(os.path.join(C.RESULTS_DIR, "preds_tpa_fullens.npy"), E)
        row = {"n_members": len(members), "tpa_full": stats(E)}
        for ref, rn in [("preds_pgf_E37", "E37"), ("preds_pgf_E18_clb", "E18_clb"),
                        ("preds_tpa_ens3", "tpa_ens3")]:
            r = L(ref.replace("preds_", ""))
            if r is not None:
                row[f"t_vs_{rn}"] = {mm: round(paired_t(E, r, y, m), 2)
                                     for mm, m in masks.items()}
                row[rn] = stats(r)
        out["tpa_fullens"] = row

    # 外部锚点（3-seed 集成级）
    if all(x is not None for x in awa3):
        A = np.mean(awa3, axis=0)
        anchors = {}
        sun3 = [L("sunset_s%d" % k) for k in range(3)]
        for nm, r in [("ConvLSTM_ens", L("convlstm_ensemble")),
                      ("SUNSET_ens", np.mean(sun3, axis=0)),
                      ("E19", L("pgf_E19_all")), ("E37", L("pgf_E37"))]:
            if r is not None:
                anchors[nm] = {mm: round(paired_t(A, r, y, m), 2)
                               for mm, m in masks.items()}
        out["anchors_t_awaens_vs"] = anchors

    with open(os.path.join(C.RESULTS_DIR, "awa_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("saved: results/awa_summary.json")


if __name__ == "__main__":
    main()
