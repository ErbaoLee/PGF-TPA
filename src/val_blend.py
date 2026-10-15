# -*- coding: utf-8 -*-
"""val 上选择两族混合权重（协议合法：§8 只允许 train/val 调参）。

族 A = 10 个 v2.1 折成员（pgf_nocsi_s{1,2}_f{k}，唯一有 ckpt 的 v2.1 成员）
族 B = 18 个 CLB 成员（pgf_nocsi_clb 固定 3 + 折 15）
共同 val = 固定划分标准 val（day_block_split seed0 val_frac0.2，stride 1）。
折模型未在标准 val 天上训练（5-fold 只切 train 天）→ 无泄漏。
流程：ckpt 重算 val+test 预测（自检 test 与已存 preds 一致）→ val 扫 w →
w* 一次性应用于 test（E28w），配对 t vs E37 / E19。输出 val_blend_summary.json。
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

MEMBERS_A = [f"pgf_nocsi_s{s}_f{k}" for s in (1, 2) for k in range(5)]
MEMBERS_B = ([f"pgf_nocsi_clb_s{s}" for s in range(3)]
             + [f"pgf_nocsi_clb_s{s}_f{k}" for s in range(3) for k in range(5)])


@torch.no_grad()
def predict(model, loader):
    model.eval()
    out = []
    for x_img, pv_log, csi_log, pt_log, pt_next, y in loader:
        out.append(model(x_img.to(DEV), pv_log.to(DEV), csi_log.to(DEV),
                         pt_next.to(DEV)).float().cpu().numpy())
    return np.concatenate(out)


def main():
    tr = C.get_split("train")
    _, va_idx = C.day_block_split(tr["anchor_time"], seed=0, val_frac=0.2)
    ds_va = C.ForecastDataset("train", va_idx, stride=1, csi=True)
    ds_te = C.ForecastDataset("test", None, stride=1, csi=True)
    kw = dict(batch_size=128, pin_memory=True, num_workers=2,
              persistent_workers=True, prefetch_factor=4)
    dl_va, dl_te = DataLoader(ds_va, shuffle=False, **kw), DataLoader(ds_te, shuffle=False, **kw)
    yv = np.concatenate([b[-1].numpy() for b in dl_va])
    te = C.get_split("test")
    yt = te["pv"][te["tgt"]].astype(np.float64)
    print(f"common val n={len(yv)}  test n={len(yt)}")

    preds = {}
    for name, clb in [(n, False) for n in MEMBERS_A] + [(n, True) for n in MEMBERS_B]:
        ck = os.path.join(C.RESULTS_DIR, f"{name}_ckpt.pt")
        model = PGF(use_csi=False, use_cl_b=clb).to(DEV)
        model.load_state_dict(torch.load(ck, map_location=DEV, weights_only=False))
        pv = predict(model, dl_va)
        pt = predict(model, dl_te)
        ref = np.load(os.path.join(C.RESULTS_DIR, f"preds_{name}.npy"))
        # conv 数值不确定性容差：逐点可差 ~1e-2 量级（float32 + cuDNN 算法选择），
        # 但整体分布必须与存档一致（差向量 RMSE 与整体 RMSE 双重校验）
        d_rmse = float(np.sqrt(np.mean((pt - ref) ** 2)))
        d_max = float(np.abs(pt - ref).max())
        assert d_rmse < 0.005 and d_max < 0.05, \
            f"{name}: ckpt/preds mismatch rmse={d_rmse} max={d_max}"
        preds[name] = (pv, pt)
        print(f"{name}: val_rmse={np.sqrt(np.mean((pv - yv) ** 2)):.4f} "
              f"test_rmse={np.sqrt(np.mean((pt - yt) ** 2)):.4f} "
              f"ckpt_check(rmse={d_rmse:.2e} max={d_max:.2e})")
        del model
        torch.cuda.empty_cache()

    A_v = np.mean([preds[n][0] for n in MEMBERS_A], axis=0)
    B_v = np.mean([preds[n][0] for n in MEMBERS_B], axis=0)
    A_t = np.mean([preds[n][1] for n in MEMBERS_A], axis=0)
    B_t = np.mean([preds[n][1] for n in MEMBERS_B], axis=0)

    ws = np.round(np.arange(0.0, 1.0001, 0.02), 2)
    curve = {float(w): float(np.sqrt(np.mean((w * A_v + (1 - w) * B_v - yv) ** 2)))
             for w in ws}
    w_star = min(curve, key=curve.get)
    print("val curve (w, val_rmse):", {w: round(r, 4) for w, r in curve.items()})
    print(f"w* = {w_star}  val_rmse(w*) = {curve[w_star]:.4f}")

    def paired_t(e1, e2):
        d = (e1 - yt) ** 2 - (e2 - yt) ** 2
        return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))

    e37 = np.load(os.path.join(C.RESULTS_DIR, "preds_pgf_E37.npy"))
    e19 = np.load(os.path.join(C.RESULTS_DIR, "preds_pgf_E19_all.npy"))
    results = {}
    for tag, p in {"E28_equal": (A_t + B_t) / 2,
                   "E28w_val": w_star * A_t + (1 - w_star) * B_t}.items():
        r = C.evaluate(yt, p, te["anchor_time"], tag=tag)
        r["skill_vs_raw_overall"] = C.forecast_skill(
            r["overall"]["rmse"], float(np.sqrt(np.mean(
                (te["pv"][te["win"]][:, -1].astype(np.float64) - yt) ** 2))))
        results[tag] = r
        np.save(os.path.join(C.RESULTS_DIR, f"preds_pgf_{tag}.npy"), p)
        print(f"{tag}: RMSE={r['overall']['rmse']:.4f} R2={r['overall']['r2']:.4f} "
              f"sunny={r['sunny']['rmse']:.4f} cloudy={r['cloudy']['rmse']:.4f}")

    p_star = w_star * A_t + (1 - w_star) * B_t
    out = {"w_star": w_star, "val_curve": curve,
           "val_rmse_A": float(np.sqrt(np.mean((A_v - yv) ** 2))),
           "val_rmse_B": float(np.sqrt(np.mean((B_v - yv) ** 2))),
           "ensembles": results,
           "paired_tests_t": {"E28w_vs_E37": paired_t(p_star, e37),
                              "E28w_vs_E19": paired_t(p_star, e19)}}
    C.dump_json(out, os.path.join(C.RESULTS_DIR, "val_blend_summary.json"))
    print(f"paired t E28w_vs_E37: {out['paired_tests_t']['E28w_vs_E37']:.2f}")
    print(f"paired t E28w_vs_E19: {out['paired_tests_t']['E28w_vs_E19']:.2f}")


if __name__ == "__main__":
    main()
