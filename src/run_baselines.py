# -*- coding: utf-8 -*-
"""Baseline 训练/评估入口（协议 v1.1，详见 BENCHMARK_LOCK.md）。

用法：
  python src/run_baselines.py --model mlp --seed 0
  python src/run_baselines.py --model sunset --seed 0 --epochs 60
协议：
  - train/val：day-block 80/20（seed0），train stride=2（官方 2-min 采样）
  - test：重建窗口全量（n=11180，晴 5697/阴 5483），永不参与训练与选择
  - 指标：RMSE/MAE/R2（sunny/cloudy/overall，kW）+ FS（分母=RAW persistence）
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from models import build  # noqa: E402


def make_loaders(args):
    tr = C.get_split("train", args.horizon)
    tr_idx, va_idx = C.day_block_split(tr["anchor_time"], seed=0, val_frac=0.2)
    no_img = args.model in ("mlp", "gru", "patchtst", "dlinear", "crossgnn")
    ds_tr = C.ForecastDataset("train", tr_idx, stride=2, no_image=no_img,
                              horizon=args.horizon)
    ds_va = C.ForecastDataset("train", va_idx, stride=2, no_image=no_img,
                              horizon=args.horizon)
    ds_te = C.ForecastDataset("test", None, stride=1, no_image=no_img,
                              horizon=args.horizon)
    workers = 0 if no_img else 4
    kw = dict(batch_size=args.batch_size, num_workers=workers, pin_memory=True,
              persistent_workers=workers > 0, prefetch_factor=4 if workers else None)
    dls = (DataLoader(ds_tr, shuffle=True, drop_last=True, **kw),
           DataLoader(ds_va, shuffle=False, **kw),
           DataLoader(ds_te, shuffle=False, **kw))
    return *dls, va_idx


@torch.no_grad()
def predict_split(model, loader, dev):
    model.eval()
    preds, ys = [], []
    for x_img, x_pv, y in loader:
        x_img, x_pv = x_img.to(dev, non_blocking=True), x_pv.to(dev, non_blocking=True)
        preds.append(model(x_img, x_pv).float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(preds), np.concatenate(ys)


def eval_test(pv_hat, args):
    """在锁定 test 上评估 + 两组 persistence 参照（horizon-aware）。"""
    te = C.get_split("test", args.horizon)
    at = te["anchor_time"]
    y_true = te["pv"][te["tgt"]].astype(np.float64)
    pv_log_te = te["pv"][te["win"]].astype(np.float64)

    res = C.evaluate(y_true, pv_hat, at,
                     tag=f"{args.model}_s{args.seed}{args.tag}")
    res["val_rmse"] = args.best_val
    res["seed"] = args.seed
    res["model"] = args.model
    res["horizon"] = args.horizon

    raw_per = pv_log_te[:, -1]
    res["raw_persistence"] = C.evaluate(y_true, raw_per, at)
    csi_per = C.csi_persistence(at, pv_log_te, y_true, horizon=args.horizon)
    res["csi_persistence"] = C.evaluate(y_true, csi_per, at)
    for k, ref in [("vs_raw", raw_per), ("vs_csi", csi_per)]:
        ref_rmse = float(np.sqrt(np.mean((ref - y_true) ** 2)))
        res[f"skill_{k}_overall"] = C.forecast_skill(res["overall"]["rmse"], ref_rmse)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["mlp", "gru", "sunset", "convlstm", "reslstm",
                             "transformer", "patchtst", "dlinear", "crossgnn"])
    ap.add_argument("--horizon", type=int, default=15, choices=[15, 5, 30],
                    help="预测尺度（分钟）。15=v1.1 锁定协议；5=v1.2；30=v1.3 扩展")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    os.makedirs(C.RESULTS_DIR, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    dl_tr, dl_va, dl_te, va_idx = make_loaders(args)
    print(f"train={len(dl_tr.dataset)} val={len(dl_va.dataset)} test={len(dl_te.dataset)}")

    model = build(args.model).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model={args.model} params={n_par/1e6:.2f}M")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    lossf = torch.nn.MSELoss()

    best_val, best_state, bad = np.inf, None, 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for x_img, x_pv, y in dl_tr:
            x_img, x_pv, y = (x_img.to(dev, non_blocking=True),
                              x_pv.to(dev, non_blocking=True),
                              y.to(dev, non_blocking=True))
            opt.zero_grad()
            loss = lossf(model(x_img, x_pv), y)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        pv_hat, pv_y = predict_split(model, dl_va, dev)
        val_rmse = float(np.sqrt(np.mean((pv_hat - pv_y) ** 2)))
        sched.step(val_rmse)
        print(f"ep{ep:03d} train_mse={tot/nb:.5f} val_rmse={val_rmse:.5f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if val_rmse < best_val - 1e-5:
            best_val, bad = val_rmse, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop at", ep)
                break

    model.load_state_dict(best_state)
    args.best_val = best_val
    pv_hat, _ = predict_split(model, dl_te, dev)
    pv_hat = pv_hat.astype(np.float64)

    res = eval_test(pv_hat, args)
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    htag = "" if args.horizon == 15 else f"_h{args.horizon}"
    base = f"{args.model}{htag}_s{args.seed}{args.tag}"
    out_json = os.path.join(C.RESULTS_DIR, base + ".json")
    out_npy = os.path.join(C.RESULTS_DIR, "preds_" + base + ".npy")
    if os.path.exists(out_json):  # 防覆盖：旧结果保留，新结果重定向 _r1/_r2...
        k = 1
        while os.path.exists(os.path.join(C.RESULTS_DIR, f"{base}_r{k}.json")):
            k += 1
        base = f"{base}_r{k}"
        out_json = os.path.join(C.RESULTS_DIR, base + ".json")
        out_npy = os.path.join(C.RESULTS_DIR, "preds_" + base + ".npy")
        print("WARN: existing result, redirecting to", out_json)
    torch.save(best_state, os.path.join(C.RESULTS_DIR, base + "_ckpt.pt"))
    C.dump_json(res, out_json)
    np.save(out_npy, pv_hat)

    # v1.3 天气场景扩展：val 天（训练未见日）stride=1 全量预测，供晴/阴/雨分日评价
    no_img = args.model in ("mlp", "gru", "patchtst", "dlinear", "crossgnn")
    ds_vf = C.ForecastDataset("train", va_idx, stride=1, no_image=no_img,
                              horizon=args.horizon)
    dl_vf = DataLoader(ds_vf, batch_size=args.batch_size, shuffle=False,
                       num_workers=0 if no_img else 4, pin_memory=True)
    pv_vf, y_vf = predict_split(model, dl_vf, dev)
    tr = C.get_split("train", args.horizon)
    np.savez(os.path.join(C.RESULTS_DIR, base + "_valfull.npz"),
             pred=pv_vf.astype(np.float32), y=y_vf.astype(np.float32),
             anchor_time=tr["anchor_time"][np.asarray(va_idx)])
    print("saved valfull:", len(pv_vf))

    print("TEST overall:", res["overall"])
    print("sunny:", res["sunny"], "\ncloudy:", res["cloudy"])


if __name__ == "__main__":
    main()
