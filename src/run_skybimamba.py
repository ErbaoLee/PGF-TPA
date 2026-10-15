# -*- coding: utf-8 -*-
"""Sky-BiMamba 官方复现训练/评估——与全部 baseline 同一锁定协议（BENCHMARK_LOCK v1.1）。

用法：
  python src/run_skybimamba.py --seed 0                 # 论文复现（纯 MSE）
  python src/run_skybimamba.py --seed 0 --ramp-w 4      # 迁移消融：+爬坡加权损失
训练协议与 run_pgf.py 完全一致（Adam 3e-4 / batch 256 / plateau / 早停 8 /
EMA 双轨评估），另加梯度裁剪 1.0（12 层 Mamba 训练稳定性，记录为协议注记）。
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
from skybimamba import SkyBiMamba  # noqa: E402


def make_loaders(args):
    tr = C.get_split("train", args.horizon)
    tr_idx, va_idx = C.day_block_split(tr["anchor_time"], seed=0, val_frac=0.2)
    ds_tr = C.ForecastDataset("train", tr_idx, stride=2, csi=True, time_feats=True,
                              horizon=args.horizon)
    ds_va = C.ForecastDataset("train", va_idx, stride=2, csi=True, time_feats=True,
                              horizon=args.horizon)
    ds_te = C.ForecastDataset("test", None, stride=1, csi=True, time_feats=True,
                              horizon=args.horizon)
    ev_bs = min(args.batch_size, args.eval_batch)  # 推理纯前向，批可大于训练微批
    kw = dict(batch_size=ev_bs, pin_memory=True, num_workers=4,
              persistent_workers=True, prefetch_factor=6)
    kw_tr = dict(kw, batch_size=args.batch_size)
    dls = (DataLoader(ds_tr, shuffle=True, drop_last=True, **kw_tr),
           DataLoader(ds_va, shuffle=False, **kw),
           DataLoader(ds_te, shuffle=False, **kw))
    return *dls, va_idx


@torch.no_grad()
def predict_split(model, loader, dev):
    model.eval()
    preds, ys = [], []
    for batch in loader:
        x_img, pv_log, csi_log, pt_log, pt_next, y, tf = batch
        preds.append(model(x_img.to(dev), pv_log.to(dev), tf.to(dev))
                     .float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(preds), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--ramp-w", type=float, default=1.0,
                    help="爬坡加权损失 λ（|ΔP|>3kW 样本权重）；迁移消融用")
    ap.add_argument("--ramp-soft", action="store_true",
                    help="软爬坡权重 w=1+(λ-1)·min(|ΔP|/3,1)（连续无硬边界）")
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--horizon", type=int, default=15, choices=[15, 5, 30],
                    help="预测尺度（分钟）。15=v1.1；5=v1.2；30=v1.3 扩展")
    ap.add_argument("--micro-batch", type=int, default=96,
                    help="物理微批（FrameStem 15帧×12通道激活大，batch 256 需"
                         "梯度累积实现；逻辑 batch 仍为 --batch-size，协议不变）")
    ap.add_argument("--amp", default="bf16", choices=["off", "bf16"],
                    help="训练混精（Ampere bf16，无 loss scale 需求；评估仍 fp32）")
    ap.add_argument("--compile", type=int, default=1,
                    help="1=尝试 torch.compile（S6 扫描 python 循环核启动瓶颈大），"
                         "失败自动回退 eager")
    ap.add_argument("--eval-batch", type=int, default=128,
                    help="评估/推理批（纯前向无梯度，可远大于训练微批；不影响结果）")
    ap.add_argument("--ema-every", type=int, default=4,
                    help="EMA 每 N 个优化器步更新一次（decay 取 N 次幂，等价平滑；提速用）")
    ap.add_argument("--ema-eval-every", type=int, default=3,
                    help="每 N 个 epoch 才额外评估一次 EMA 权重（其余 epoch 仅 raw；提速用）")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    suffix = ""
    if args.ramp_w != 1.0:
        suffix = ("_rws%g" if args.ramp_soft else "_rw%g") % args.ramp_w
    name = "skybimamba" + suffix + ("" if args.horizon == 15 else f"_h{args.horizon}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    dl_tr, dl_va, dl_te, va_idx = make_loaders(args)
    print(f"{name} train={len(dl_tr.dataset)} val={len(dl_va.dataset)} "
          f"test={len(dl_te.dataset)}")

    model = SkyBiMamba(d_model=args.d_model, n_layers=args.layers).to(dev)
    raw_model = model  # compile 后 state_dict 带 _orig_mod 前缀；权重存取一律走 raw
    print(f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    if args.compile:
        # 整模型编译在 PVEmbed 索引算术上失败（inductor ModularIndexing）；
        # 只编译瓶颈 MambaBlock（S6 扫描 python 循环），先临时块预热验证。
        try:
            from skybimamba import MambaBlock
            scratch = MambaBlock(args.d_model).to(dev)
            xs = torch.randn(2, 15, args.d_model, device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cs = torch.compile(scratch)
                cs(xs).float().sum().backward()
            del scratch, cs, xs
            n = 0
            for stack in (raw_model.img_mamba, raw_model.pv_mamba):
                for layer in stack:
                    layer.fwd = torch.compile(layer.fwd)
                    layer.bwd = torch.compile(layer.bwd)
                    n += 2
            print(f"torch.compile enabled on {n} MambaBlocks")
        except Exception as ex:
            print("torch.compile unusable, eager fallback:", ex)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)

    def batch_loss(yhat, y, pv_log):
        if args.ramp_w != 1.0:
            dp = (y - pv_log[:, -1]).abs()
            if args.ramp_soft:
                w = 1.0 + (args.ramp_w - 1.0) * torch.clamp(dp / 3.0, max=1.0)
            else:
                w = 1.0 + (args.ramp_w - 1.0) * (dp > 3.0).float()
            return (((yhat - y) ** 2) * w).mean()
        return torch.nn.functional.mse_loss(yhat, y)

    best_val, best_state, bad = np.inf, None, 0
    best_src = "raw"
    ema = {k: v.detach().clone().float() for k, v in raw_model.state_dict().items()
           if v.dtype.is_floating_point}
    ema_decay = 0.999

    def load_into(dst, src):
        dst.load_state_dict(src, strict=False)

    t0 = time.time()
    gstep = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for batch in dl_tr:
            x_img, pv_log, csi_log, pt_log, pt_next, y, tf = batch
            n = x_img.shape[0]
            n_chunk = max(1, (n + args.micro_batch - 1) // args.micro_batch)
            opt.zero_grad()
            loss_sum = 0.0
            for cx, cpv, cy, ctf in zip(x_img.split(args.micro_batch),
                                         pv_log.split(args.micro_batch),
                                         y.split(args.micro_batch),
                                         tf.split(args.micro_batch)):
                cx, cpv, cy, ctf = (cx.to(dev), cpv.to(dev), cy.to(dev), ctf.to(dev))
                if args.amp == "bf16":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        yhat = model(cx, cpv, ctf)
                        loss = batch_loss(yhat.float(), cy, cpv) / n_chunk
                else:
                    yhat = model(cx, cpv, ctf)
                    loss = batch_loss(yhat, cy, cpv)
                loss.backward()
                loss_sum += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            gstep += 1
            if gstep % args.ema_every == 0:  # 每 N 步更新一次（decay 取 N 次幂，等价平滑）
                m = ema_decay ** args.ema_every
                with torch.no_grad():
                    sd_now = raw_model.state_dict()
                    for k, v in ema.items():
                        v.mul_(m).add_(sd_now[k].float(), alpha=1 - m)
            tot += loss_sum
            nb += 1
        pv_hat, pv_y = predict_split(model, dl_va, dev)
        val_raw = float(np.sqrt(np.mean((pv_hat - pv_y) ** 2)))
        if ep % args.ema_eval_every == 0 or ep == 1:  # EMA 评估降频（省一半推理）
            sd_backup = {k: v.detach().clone() for k, v in raw_model.state_dict().items()}
            load_into(raw_model, ema)
            pv_hat_e, _ = predict_split(model, dl_va, dev)
            val_ema = float(np.sqrt(np.mean((pv_hat_e - pv_y) ** 2)))
            load_into(raw_model, sd_backup)
        else:
            val_ema = np.inf
        val_rmse, src = (val_raw, "raw") if val_raw <= val_ema else (val_ema, "ema")
        sched.step(val_rmse)
        print(f"ep{ep:03d} train_mse={tot/nb:.5f} val_rmse={val_rmse:.5f} "
              f"[raw {val_raw:.5f} | ema {val_ema:.5f}] ({time.time()-t0:.0f}s)",
              flush=True)
        if val_rmse < best_val - 1e-5:
            best_val, bad, best_src = val_rmse, 0, src
            if src == "ema":
                load_into(raw_model, ema)
                best_state = {k: v.detach().clone()
                              for k, v in raw_model.state_dict().items()}
                load_into(raw_model, sd_backup)
            else:
                best_state = {k: v.detach().clone()
                              for k, v in raw_model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop at", ep)
                break

    raw_model.load_state_dict(best_state)
    pv_hat, _ = predict_split(model, dl_te, dev)
    pv_hat = pv_hat.astype(np.float64)

    te = C.get_split("test", args.horizon)
    y_true = te["pv"][te["tgt"]].astype(np.float64)
    res = C.evaluate(y_true, pv_hat, te["anchor_time"], tag=f"{name}_s{args.seed}")
    res.update({"val_rmse": best_val, "seed": args.seed, "model": name,
                "best_src": best_src, "horizon": args.horizon})
    raw_per = te["pv"][te["win"]][:, -1].astype(np.float64)
    res["skill_vs_raw_overall"] = C.forecast_skill(
        res["overall"]["rmse"], float(np.sqrt(np.mean((raw_per - y_true) ** 2))))

    out_json = os.path.join(C.RESULTS_DIR, f"{name}_s{args.seed}{args.tag}.json")
    out_npy = os.path.join(C.RESULTS_DIR, f"preds_{name}_s{args.seed}{args.tag}.npy")
    if os.path.exists(out_json):
        k = 1
        while os.path.exists(os.path.join(
                C.RESULTS_DIR, f"{name}_s{args.seed}{args.tag}_r{k}.json")):
            k += 1
        out_json = os.path.join(C.RESULTS_DIR, f"{name}_s{args.seed}{args.tag}_r{k}.json")
        out_npy = out_json.replace(".json", ".npy").replace("results_", "preds_")
        print("WARN: existing result, redirecting to", out_json)
    stem = os.path.basename(out_json)[:-5]
    torch.save(best_state, os.path.join(C.RESULTS_DIR, stem + "_ckpt.pt"))
    C.dump_json(res, out_json)
    np.save(out_npy, pv_hat)

    # v1.3 天气场景扩展：val 天（训练未见日）stride=1 全量预测
    ds_vf = C.ForecastDataset("train", va_idx, stride=1, csi=True, time_feats=True,
                              horizon=args.horizon)
    from torch.utils.data import DataLoader as _DL
    dl_vf = _DL(ds_vf, batch_size=args.eval_batch, shuffle=False, num_workers=2,
                pin_memory=True)
    pv_vf, y_vf = predict_split(model, dl_vf, dev)
    tr_sp = C.get_split("train", args.horizon)
    np.savez(os.path.join(C.RESULTS_DIR, stem + "_valfull.npz"),
             pred=pv_vf.astype(np.float32), y=y_vf.astype(np.float32),
             anchor_time=tr_sp["anchor_time"][np.asarray(va_idx)])
    print("saved valfull:", len(pv_vf))

    print("TEST overall:", res["overall"])
    print("sunny:", res["sunny"], "\ncloudy:", res["cloudy"])


if __name__ == "__main__":
    main()
