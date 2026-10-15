# -*- coding: utf-8 -*-
"""Nowcast 预训练 B 分支编码器（阶段二实验 2）。

任务：15 帧窗口 [t-14,t] → 末帧时刻 PV（=pv_log[-1]）。目标在窗口末端，
不含任何未来信息；训练只用 trainval 天（fold0 的 train 部分，与主训练同
划分口径），test 天永不参与。输入端遮蔽 pv_log 末 3 值（见 BBranch），
迫使图像分支真正学习图像→功率映射，而非走"输入含目标"的恒等捷径。
预训练后 B 分支获得图像→功率的校准映射，再在 forecast 任务上微调
（run_pgf.py --pretrain）。
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
from models import PGF, frames_to_channels  # noqa: E402


class BBranch(torch.nn.Module):
    """独立包装 PGF 的 B 分支（conv+enc_b+head_b），键名与 PGF 一致以便回载。

    输入端将 pv_log 末 3 个值置零：目标恰是 pv_log[-1]，若不遮蔽，
    网络会走恒等捷径（输入直接含目标），图像分支学不到任何东西。
    置零后模型必须从天空图像 + 前 12 个 PV 值外推当前功率——这正是
    我们希望预训练注入的"图像→功率校准映射"。
    """

    def __init__(self):
        super().__init__()
        tmp = PGF(use_gate=False)  # 只为借用模块定义
        self.conv = tmp.conv
        self.enc_b = tmp.enc_b
        self.head_b = tmp.head_b

    def forward(self, x_img, pv_log):
        pv_in = pv_log.clone()
        pv_in[:, -3:] = 0.0
        x = self.conv(frames_to_channels(x_img))
        x = torch.flatten(x, 1)
        h = torch.relu(self.enc_b(torch.cat([x, pv_in], dim=1)))
        return self.head_b(h).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/proc/nowcast_pretrain.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr = C.get_split("train")
    tr_idx, va_idx = C.day_block_split(tr["anchor_time"], seed=0, val_frac=0.2)
    # nowcast 预训练同样用 80% 天训练 / 20% 早停（与主实验同口径）
    ds_tr = C.ForecastDataset("train", tr_idx, stride=2, csi=False, target_nowcast=True)
    ds_va = C.ForecastDataset("train", va_idx, stride=2, csi=False, target_nowcast=True)
    kw = dict(batch_size=args.batch_size, pin_memory=True, num_workers=2,
              persistent_workers=True, prefetch_factor=4)
    dl_tr = DataLoader(ds_tr, shuffle=True, drop_last=True, **kw)
    dl_va = DataLoader(ds_va, shuffle=False, **kw)
    print(f"nowcast-pretrain train={len(ds_tr)} val={len(ds_va)}")

    model = BBranch().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = torch.nn.MSELoss()

    @torch.no_grad()
    def val_rmse():
        model.eval()
        ps, ys = [], []
        for x_img, x_pv, y in dl_va:
            ps.append(model(x_img.to(dev), x_pv.to(dev)).float().cpu().numpy())
            ys.append(y.numpy())
        return float(np.sqrt(np.mean((np.concatenate(ps) - np.concatenate(ys)) ** 2)))

    best, best_state = np.inf, None
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for x_img, x_pv, y in dl_tr:
            x_img, x_pv, y = x_img.to(dev), x_pv.to(dev), y.to(dev)
            opt.zero_grad()
            loss = lossf(model(x_img, x_pv), y)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        v = val_rmse()
        print(f"ep{ep:03d} train_mse={tot/nb:.5f} nowcast_val_rmse={v:.5f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if v < best - 1e-5:
            best, best_state = v, {k: t.detach().clone() for k, t in model.state_dict().items()}

    torch.save(best_state, args.out)
    print(f"saved pretrain weights -> {args.out} (best nowcast val rmse {best:.4f})")


if __name__ == "__main__":
    main()
