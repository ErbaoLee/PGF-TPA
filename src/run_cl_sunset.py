# -*- coding: utf-8 -*-
"""CL-SUNSET 复现（IEEE IAS 2025, Lu/Chen/Hsieh；机制级复现）。

机制：SimCLR 自监督对比预训练 SUNSET 的 conv 编码器（只用 train 天图像，
无标签、无 test 信息），随后按锁定协议微调 SUNSET 回归头。
与 run_nowcast_pretrain 的监督式预训练（负结果）形成对照：对比学习的
不变性目标不依赖"图像→功率"的捷径，可能学到更可迁移的云表征。

原文付费墙，增强/温度/投影头为标准 SimCLR 配置（记录为实现假设）：
  RandomResizedCrop(0.5-1.0) + ColorJitter(b0.4,c0.4,s0.2) + 灰度(0.1)
  + 高斯模糊(0.5)；NT-Xent τ=0.5；投影头 flat→256→128。

用法：
  python src/run_cl_sunset.py                # 预训练（保存 cl_sunset_encoder.pt）
  python src/run_cl_sunset.py --ft --seed 0  # 微调 3 seeds
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from models import SunsetCNN  # noqa: E402


class FrameDataset(Dataset):
    """train 天的逐分钟帧（取非重叠窗口的末帧，stride=15 → 帧不重复）。"""

    def __init__(self, idx):
        d = C.get_split("train")
        self.d = d
        # 每 15 个窗口锚点取一次末帧 → 分钟粒度近似不重叠
        rows = d["win"][idx][:, -1]
        self.rows = np.unique(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        arr = np.asarray(self.d["images"][self.rows[i]])
        img = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1)
        return img

    def __getstate__(self):
        st = self.__dict__.copy()
        st.pop("d", None)
        return st

    def __setstate__(self, st):
        self.__dict__.update(st)
        self.d = C.get_split("train")  # Windows spawn 懒重建


class TwoViews(Dataset):
    def __init__(self, base, tf):
        self.base = base
        self.tf = tf

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        return self.tf(self.base[i]), self.tf(self.base[i])


class SimCLR(nn.Module):
    """单帧（3 通道）SUNSET 风格编码器 + 投影头。

    与 SunsetCNN.conv 层索引对齐（0=Conv24,1=BN,4=Conv48,5=BN），
    微调时首层 (24,3,3,3) 按帧复制为 (24,45,3,3)（视频模型 ImageNet
    初始化的标准做法，除以 sqrt(15) 保持激活量级），其余层直接回载。
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(24, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.proj = nn.Sequential(
            nn.Linear(48 * 16 * 16, 256), nn.ReLU(), nn.Linear(256, 128))

    def forward(self, x):
        h = self.conv(x)
        return F.normalize(self.proj(torch.flatten(h, 1)), dim=1)


def nt_xent(z1, z2, tau=0.5):
    z = torch.cat([z1, z2], 0)
    sim = z @ z.t() / tau
    n = z1.shape[0]
    sim.fill_diagonal_(-1e9)
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


def pretrain(args):
    torch.manual_seed(0)
    np.random.seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = C.get_split("train")
    tr_idx, va_idx = C.day_block_split(tr["anchor_time"], seed=0, val_frac=0.2)
    # torchvision transforms
    import torchvision.transforms as T
    tf = T.Compose([
        T.RandomResizedCrop(64, scale=(0.5, 1.0)),
        T.ColorJitter(0.4, 0.4, 0.2, 0.0),
        T.RandomGrayscale(p=0.1),
        T.GaussianBlur(3, sigma=(0.1, 2.0)),
    ])
    ds_tr = TwoViews(FrameDataset(tr_idx), tf)
    ds_va = FrameDataset(va_idx)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, drop_last=True,
                       num_workers=2, pin_memory=True, persistent_workers=True,
                       prefetch_factor=4)
    dl_va = DataLoader(ds_va, batch_size=args.batch, num_workers=2, pin_memory=True)
    print(f"contrastive frames train={len(ds_tr)} val={len(ds_va)}")

    model = SimCLR().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    best, best_state = np.inf, None
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for v1, v2 in dl_tr:
            v1, v2 = v1.to(dev), v2.to(dev)
            opt.zero_grad()
            loss = nt_xent(model(v1), model(v2))
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()
        # val NT-Xent 监控（同视图对无意义 → 用固定弱增强对）
        model.eval()
        with torch.no_grad():
            vt, vn = 0.0, 0
            for img in dl_va:
                img = img.to(dev)
                v1 = T.functional.adjust_contrast(img, 1.2)
                v2 = T.functional.adjust_saturation(img, 0.8)
                vt += nt_xent(model(v1), model(v2)).item()
                vn += 1
        v = vt / max(vn, 1)
        print(f"ep{ep:03d} train_ntxent={tot/nb:.4f} val_ntxent={v:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if v < best - 1e-4:
            best, best_state = v, {k: t.detach().clone()
                                   for k, t in model.conv.state_dict().items()}
    torch.save(best_state, args.out)
    print(f"saved encoder -> {args.out} (best val ntxent {best:.4f})")


def finetune(args):
    """加载预训练 conv，按锁定协议微调完整 SUNSET（复用 run_baselines 主流程）。"""
    import run_baselines as RB

    class Args:
        pass
    a = Args()
    a.model, a.seed, a.epochs = "cl_sunset", args.seed, args.epochs_ft
    a.batch_size, a.lr, a.patience, a.tag = args.batch_ft, 3e-4, 8, ""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dl_tr, dl_va, dl_te = RB.make_loaders(a)
    model = SunsetCNN().to(dev)
    sd = torch.load(args.out, map_location=dev)
    load = {}
    for k, v in sd.items():
        if k == "0.weight":  # (24,3,3,3) -> (24,45,3,3) 帧复制初始化
            load["conv.0.weight"] = v.repeat(1, 15, 1, 1) / np.sqrt(15.0)
        else:
            load["conv." + k] = v
    missing, unexpected = model.load_state_dict(load, strict=False)
    print(f"pretrained conv loaded: missing={len(missing)} unexpected={len(unexpected)}")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    lossf = nn.MSELoss()
    best_val, best_state, bad = np.inf, None, 0
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
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
        pv_hat, pv_y = RB.predict_split(model, dl_va, dev)
        val_rmse = float(np.sqrt(np.mean((pv_hat - pv_y) ** 2)))
        sched.step(val_rmse)
        print(f"ep{ep:03d} train_mse={tot/nb:.5f} val_rmse={val_rmse:.5f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if val_rmse < best_val - 1e-5:
            best_val, bad = val_rmse, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                print("early stop at", ep)
                break
    model.load_state_dict(best_state)
    a.best_val = best_val
    pv_hat, _ = RB.predict_split(model, dl_te, dev)
    res = RB.eval_test(pv_hat.astype(np.float64), a)
    C.dump_json(res, os.path.join(C.RESULTS_DIR, f"cl_sunset_s{args.seed}.json"))
    np.save(os.path.join(C.RESULTS_DIR, f"preds_cl_sunset_s{args.seed}.npy"),
            pv_hat.astype(np.float64))
    print("TEST overall:", res["overall"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default="data/proc/cl_sunset_encoder.pt")
    ap.add_argument("--ft", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs-ft", type=int, default=60)
    ap.add_argument("--batch-ft", type=int, default=256)
    args = ap.parse_args()
    if args.ft:
        finetune(args)
    else:
        pretrain(args)


if __name__ == "__main__":
    main()
