# -*- coding: utf-8 -*-
"""PGF（Physics-Gated Fusion）训练/评估——与 baseline 完全同一锁定协议。

用法：
  python src/run_pgf.py --seed 0                      # 完整 PGF
  python src/run_pgf.py --seed 0 --no-csi             # 消融：去物理参数化
  python src/run_pgf.py --seed 0 --no-gate            # 消融：固定 0.5 融合
  python src/run_pgf.py --seed 0 --no-image           # 消融：去图像分支
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
from models import PGF  # noqa: E402


def make_loaders(args):
    tr = C.get_split("train", args.horizon)
    if args.fold >= 0:
        splits = list(C.day_block_kfold(tr["anchor_time"], n_folds=args.n_folds, seed=0))
        tr_idx, va_idx = splits[args.fold][0], splits[args.fold][1]
    else:
        tr_idx, va_idx = C.day_block_split(tr["anchor_time"], seed=0, val_frac=0.2)
    ds_tr = C.ForecastDataset("train", tr_idx, stride=2, csi=True,
                              no_image=args.no_image, horizon=args.horizon)
    ds_va = C.ForecastDataset("train", va_idx, stride=2, csi=True,
                              no_image=args.no_image, horizon=args.horizon)
    ds_te = C.ForecastDataset("test", None, stride=1, csi=True,
                              no_image=args.no_image, horizon=args.horizon)
    workers = 0 if args.no_image else 4
    kw = dict(batch_size=args.batch_size, pin_memory=True, num_workers=workers,
              persistent_workers=workers > 0,
              prefetch_factor=4 if workers else None)
    dls = (DataLoader(ds_tr, shuffle=True, drop_last=True, **kw),
           DataLoader(ds_va, shuffle=False, **kw),
           DataLoader(ds_te, shuffle=False, **kw))
    return *dls, va_idx


@torch.no_grad()
def predict_split(model, loader, dev, residual=False):
    model.eval()
    preds, ys = [], []
    for batch in loader:
        x_img, pv_log, csi_log, pt_log, pt_next, y = batch
        out = model(x_img.to(dev), pv_log.to(dev), csi_log.to(dev),
                    pt_next.to(dev)).float()
        if residual:  # ŷ = P(t) + Δ̂
            out = out + pv_log[:, -1].to(dev).float()
        preds.append(out.cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(preds), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--no-csi", action="store_true")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--no-image", action="store_true")
    ap.add_argument("--fd", action="store_true")
    ap.add_argument("--rel-loss", action="store_true",
                    help="v4: per-sample scale=sqrt(1+var(pv_log)) 的相对误差损失")
    ap.add_argument("--deep", action="store_true",
                    help="B 分支 ResNet 式加深（v5）")
    ap.add_argument("--cl-b", action="store_true",
                    help="B 分支换 ConvLSTM 时空编码（阶段三，源自官方 Cambridge 架构复现）")
    ap.add_argument("--awa-b", action="store_true",
                    help="B 分支换 AWA 平流外推聚合编码（阶段四，自创：运动场 warp 到 t+15 参考系）")
    ap.add_argument("--awa-noalign", action="store_true",
                    help="AWA 消融：位移场置零（纯时间平均聚合，检验对齐必要性）")
    ap.add_argument("--awa-noextrap", action="store_true",
                    help="AWA 消融：warp 到 t=14 当前系而非 t+15（检验外推必要性）")
    ap.add_argument("--awa-near0", action="store_true",
                    help="AWA 消融：去近帧状态流（检验双流设计的近帧必要性）")
    ap.add_argument("--tpa", action="store_true",
                    help="TPA 编码器（阶段四终版自创：帧间共享深空间编码 + 全时程"
                         "时间池化读出，无递归无warp；= --awa-b --awa-near0 --awa-noalign）")
    ap.add_argument("--tpa-ms", action="store_true",
                    help="L3a：TPA 多尺度读出 mean‖last3mean‖last‖trend"
                         "（爬坡短时趋势显式通道，需配合 --tpa）")
    ap.add_argument("--tpa-sa", action="store_true",
                    help="L3b：TPA 空间注意力读出（末帧可学习空间 softmax 聚焦池化，"
                         "保留日盘/热点邻域局部信息，需配合 --tpa）")
    ap.add_argument("--gate-oracle", type=float, default=0.0,
                    help="v6: gate 的 oracle 择优蒸馏辅助损失权重（如 0.3）")
    ap.add_argument("--ramp-w", type=float, default=1.0,
                    help="爬坡感知加权：|y-PV(t)|>3kW(10%%容量) 样本损失权重（如 4）")
    ap.add_argument("--a-aux", type=float, default=0.0,
                    help="A 分支独立预测辅助损失权重 β（阶段四：AWA 版门控 g 更高"
                         "→A 梯度更弱退化更狠，B=13.2 vs CLB 版 12.1；直接补 A 训练信号）")
    ap.add_argument("--fold", type=int, default=-1,
                    help=">=0: 官方同款 day-block K 折的第 k 折作验证集")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--pretrain", default="",
                    help="nowcast 预训练权重路径（加载到 B 分支）")
    ap.add_argument("--tag", default="")
    ap.add_argument("--residual", action="store_true",
                    help="v2 配方：目标改为持久性残差 ΔP=y-P(t)，预测 ŷ=P(t)+Δ̂"
                         "（短尺度晴天平滑曲线的时滞/过平滑矫正）")
    ap.add_argument("--l1-w", type=float, default=0.0,
                    help="v2 配方：损失 = 原损失 + λ·L1（λ 如 0.5，直击 MAE）")
    ap.add_argument("--horizon", type=int, default=15, choices=[15, 5, 30],
                    help="预测尺度（分钟）。15=v1.1 锁定协议；5=v1.2；30=v1.3 扩展")
    args = ap.parse_args()
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    if args.tpa:  # TPA = AWA 消融矩阵收敛的终版（warp/近帧流均已证无贡献）
        args.awa_b = True
        args.awa_near0 = True
        args.awa_noalign = True
    name = "pgf" + ("_nogate" if args.no_gate else "") + \
           ("_nocsi" if args.no_csi else "") + ("_noimg" if args.no_image else "") + \
           ("_fd" if args.fd else "") + ("_rel" if args.rel_loss else "") + \
           ("_deep" if args.deep else "") + ("_clb" if args.cl_b else "") + \
           ("_tpa" if args.tpa else "") + ("_tpams" if args.tpa_ms else "") + \
           ("_tpasa" if args.tpa_sa else "") + \
           ("" if args.tpa else ("_awa" if args.awa_b else "")) + \
           ("" if args.tpa or not args.awa_b else
            ("_awana" if args.awa_noalign else "") +
            ("_awane" if args.awa_noextrap else "") +
            ("_awanr" if args.awa_near0 else "")) + \
           ("_go%.1f" % args.gate_oracle if args.gate_oracle > 0 else "") + \
           ("_rw%g" % args.ramp_w if args.ramp_w != 1.0 else "") + \
           ("_aa%g" % args.a_aux if args.a_aux > 0 else "") + \
           ("_res" if args.residual else "") + \
           ("_l1%g" % args.l1_w if args.l1_w > 0 else "") + \
           ("" if args.horizon == 15 else f"_h{args.horizon}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    dl_tr, dl_va, dl_te, va_idx = make_loaders(args)
    print(f"{name} train={len(dl_tr.dataset)} val={len(dl_va.dataset)} test={len(dl_te.dataset)}")

    model = PGF(use_csi=not args.no_csi, use_gate=not args.no_gate,
                use_image=not args.no_image, use_fd=args.fd,
                use_deep=args.deep, use_cl_b=args.cl_b,
                use_awa_b=args.awa_b, awa_align=not args.awa_noalign,
                awa_extrap=not args.awa_noextrap,
                awa_near=not args.awa_near0, tpa_ms=args.tpa_ms,
                tpa_sa=args.tpa_sa).to(dev)
    if args.pretrain:
        import torch as _t
        sd = _t.load(args.pretrain, map_location=dev)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"pretrain loaded: missing={len(missing)} unexpected={len(unexpected)}")
    print(f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    lossf = torch.nn.MSELoss()

    best_val, best_state, bad = np.inf, None, 0
    best_src = "raw"
    # EMA 权重影子：训练过程改进（非架构改动）。每个 epoch 同时评估
    # raw/EMA 两套权重，val 更优者参与早停与最终选择——选择只依赖 val，无泄漏。
    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()
           if v.dtype.is_floating_point}
    ema_decay = 0.999

    def load_into(dst, src):
        dst.load_state_dict(src, strict=False)

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for batch in dl_tr:
            x_img, pv_log, csi_log, pt_log, pt_next, y = batch
            x_img, pv_log, csi_log = (x_img.to(dev), pv_log.to(dev), csi_log.to(dev))
            pt_next, y = pt_next.to(dev), y.to(dev)
            opt.zero_grad()
            if args.gate_oracle > 0 or args.a_aux > 0:
                yhat, g, f_a_d, f_b_d = model(x_img, pv_log, csi_log, pt_next,
                                              return_parts=True)
            else:
                yhat = model(x_img, pv_log, csi_log, pt_next)
            y_eff = (y - pv_log[:, -1]) if args.residual else y
            if args.rel_loss:
                scale = torch.sqrt(1.0 + pv_log.var(dim=1))
                loss = lossf((yhat - y_eff) / scale, torch.zeros_like(yhat))
            elif args.ramp_w != 1.0:
                w = 1.0 + (args.ramp_w - 1.0) * ((y - pv_log[:, -1]).abs() > 3.0).float()
                loss = (((yhat - y_eff) ** 2) * w).mean()
            else:
                loss = lossf(yhat, y_eff)
            if args.l1_w > 0:
                loss = loss + args.l1_w * torch.nn.functional.l1_loss(yhat, y_eff)
            if args.a_aux > 0:
                loss = loss + args.a_aux * lossf(f_a_d, y_eff)
            if args.gate_oracle > 0:
                target = ((y - f_a_d).abs() > (y - f_b_d).abs()).float()
                bce = torch.nn.functional.binary_cross_entropy(
                    torch.clamp(g, 1e-4, 1 - 1e-4), target)
                loss = loss + args.gate_oracle * bce
            loss.backward()
            opt.step()
            with torch.no_grad():
                sd_now = model.state_dict()
                for k, v in ema.items():
                    v.mul_(ema_decay).add_(sd_now[k].float(), alpha=1 - ema_decay)
            tot += loss.item()
            nb += 1
        # raw 权重评估
        pv_hat, pv_y = predict_split(model, dl_va, dev, args.residual)
        val_raw = float(np.sqrt(np.mean((pv_hat - pv_y) ** 2)))
        # EMA 权重评估（BN 统计量沿用当前值）
        sd_backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        load_into(model, ema)
        pv_hat_e, _ = predict_split(model, dl_va, dev, args.residual)
        val_ema = float(np.sqrt(np.mean((pv_hat_e - pv_y) ** 2)))
        load_into(model, sd_backup)
        val_rmse, src = (val_raw, "raw") if val_raw <= val_ema else (val_ema, "ema")
        sched.step(val_rmse)
        print(f"ep{ep:03d} train_mse={tot/nb:.5f} val_rmse={val_rmse:.5f} "
              f"[raw {val_raw:.5f} | ema {val_ema:.5f}] ({time.time()-t0:.0f}s)",
              flush=True)
        if val_rmse < best_val - 1e-5:
            best_val, bad, best_src = val_rmse, 0, src
            if src == "ema":
                load_into(model, ema)
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                load_into(model, sd_backup)
            else:
                best_state = {k: v.detach().clone() for k, v in sd_backup.items()}
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop at", ep)
                break

    model.load_state_dict(best_state)
    pv_hat_va, pv_y_va = predict_split(model, dl_va, dev, args.residual)
    pv_hat, _ = predict_split(model, dl_te, dev, args.residual)
    pv_hat = pv_hat.astype(np.float64)

    te = C.get_split("test", args.horizon)
    at = te["anchor_time"]
    y_true = te["pv"][te["tgt"]].astype(np.float64)
    res = C.evaluate(y_true, pv_hat, at, tag=f"{name}_s{args.seed}{args.tag}")
    res.update({"val_rmse": best_val, "seed": args.seed, "model": name,
                "best_src": best_src, "horizon": args.horizon,
                "ablation": {"csi": not args.no_csi, "gate": not args.no_gate,
                             "image": not args.no_image}})
    raw_per = te["pv"][te["win"]][:, -1].astype(np.float64)
    res["skill_vs_raw_overall"] = C.forecast_skill(
        res["overall"]["rmse"], float(np.sqrt(np.mean((raw_per - y_true) ** 2))))

    out_json = os.path.join(C.RESULTS_DIR, f"{name}_s{args.seed}{args.tag}.json")
    out_npy = os.path.join(C.RESULTS_DIR, f"preds_{name}_s{args.seed}{args.tag}.npy")
    if os.path.exists(out_json):
        k = 1
        while os.path.exists(os.path.join(C.RESULTS_DIR, f"{name}_s{args.seed}{args.tag}_r{k}.json")):
            k += 1
        out_json = os.path.join(C.RESULTS_DIR, f"{name}_s{args.seed}{args.tag}_r{k}.json")
        out_npy = os.path.join(C.RESULTS_DIR, "preds_" + os.path.basename(out_json).replace(".json", ".npy"))
        print("WARN: existing result, redirecting to", out_json)
    stem = os.path.basename(out_json)[:-5]
    torch.save(best_state, os.path.join(C.RESULTS_DIR, stem + "_ckpt.pt"))
    np.savez(os.path.join(C.RESULTS_DIR, stem + "_val.npz"),
             val_pred=pv_hat_va.astype(np.float32), val_y=pv_y_va.astype(np.float32))
    # v1.3 天气场景扩展：val 天（训练未见日）stride=1 全量预测，供晴/阴/雨分日评价
    ds_vf = C.ForecastDataset("train", va_idx, stride=1, csi=True,
                              no_image=args.no_image, horizon=args.horizon)
    from torch.utils.data import DataLoader as _DL
    dl_vf = _DL(ds_vf, batch_size=args.batch_size, shuffle=False,
                num_workers=0 if args.no_image else 4, pin_memory=True)
    pv_vf, y_vf = predict_split(model, dl_vf, dev, args.residual)
    tr_sp = C.get_split("train", args.horizon)
    np.savez(os.path.join(C.RESULTS_DIR, stem + "_valfull.npz"),
             pred=pv_vf.astype(np.float32), y=y_vf.astype(np.float32),
             anchor_time=tr_sp["anchor_time"][np.asarray(va_idx)])
    print("saved valfull:", len(pv_vf))
    C.dump_json(res, out_json)
    np.save(out_npy, pv_hat)
    print("TEST overall:", res["overall"])
    print("sunny:", res["sunny"], "\ncloudy:", res["cloudy"])


if __name__ == "__main__":
    main()
