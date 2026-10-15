# -*- coding: utf-8 -*-
"""v1.3：为已训练完成的模型 checkpoint 回填 val 天 stride=1 全量预测
（{stem}_valfull.npz），供天气场景扩展表（晴/阴/雨）使用。

支持：
  - run_baselines 家族（ckpt 结构 = model.state_dict()）
  - run_pgf 家族（从 json 的 model 名解析架构旗标）
  - skybimamba（时间编码输入）
用法：
  python src/backfill_valfull.py pgf_nocsi_tpa_rw4 15
  python src/backfill_valfull.py sunset 15        # 需 _ckpt.pt 存在
  python src/backfill_valfull.py skybimamba 30
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from models import build, PGF  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def build_pgf(name):
    """从结果名解析 PGF 架构旗标（与 run_pgf.py 的命名规则一致）。"""
    kw = dict(use_csi="_nocsi" not in name, use_gate="_nogate" not in name,
              use_image="_noimg" not in name,
              use_fd="_fd" in name, use_deep="_deep" in name,
              use_cl_b="_clb" in name, use_awa_b=("_awa" in name or "_tpa" in name),
              awa_align=not ("_awana" in name or "_tpa" in name),
              awa_extrap=not ("_awane" in name),
              awa_near=not ("_awanr" in name or "_tpa" in name),
              tpa_ms="_tpams" in name, tpa_sa="_tpasa" in name)
    return PGF(**kw), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")           # 结果文件名 stem（不含 _s{seed}）
    ap.add_argument("horizon", type=int)
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()

    for s in [int(x) for x in args.seeds.split(",")]:
        stem = f"{args.model}_s{s}"
        ck = os.path.join(C.RESULTS_DIR, stem + "_ckpt.pt")
        out = os.path.join(C.RESULTS_DIR, stem + "_valfull.npz")
        if not os.path.exists(ck):
            print("MISS ckpt:", ck)
            continue
        if os.path.exists(out):
            print("SKIP exists:", out)
            continue

        if args.model.startswith("pgf"):
            model, is_pgf = build_pgf(args.model)
        elif args.model.startswith("skybimamba"):
            from skybimamba import SkyBiMamba
            model = SkyBiMamba()
            is_pgf = False
        else:
            model = build(args.model)
            is_pgf = False
        from infer import load_checkpoint
        sd = load_checkpoint(ck)
        model.load_state_dict(sd)
        model.to(DEV).eval()

        tr = C.get_split("train", args.horizon)
        _, va_idx = C.day_block_split(tr["anchor_time"], seed=0, val_frac=0.2)
        if is_pgf:
            ds = C.ForecastDataset("train", va_idx, stride=1, csi=True,
                                   horizon=args.horizon)
        elif args.model.startswith("skybimamba"):
            ds = C.ForecastDataset("train", va_idx, stride=1, csi=True,
                                   time_feats=True, horizon=args.horizon)
        else:
            no_img = args.model in ("mlp", "gru", "patchtst", "dlinear", "crossgnn")
            ds = C.ForecastDataset("train", va_idx, stride=1, csi=False,
                                   no_image=no_img, horizon=args.horizon)
        dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4,
                        pin_memory=True)
        preds, ys = [], []
        with torch.no_grad():
            for batch in dl:
                if is_pgf:
                    x_img, pv_log, csi_log, pt_log, pt_next, y = batch
                    p = model(x_img.to(DEV), pv_log.to(DEV), csi_log.to(DEV),
                              pt_next.to(DEV))
                elif args.model.startswith("skybimamba"):
                    x_img, pv_log, csi_log, pt_log, pt_next, y, tf = batch
                    p = model(x_img.to(DEV), pv_log.to(DEV), tf.to(DEV))
                else:
                    x_img, pv_log, y = batch
                    p = model(x_img.to(DEV), pv_log.to(DEV))
                preds.append(p.float().cpu().numpy())
                ys.append(y.numpy())
        preds = np.concatenate(preds).astype(np.float32)
        ys = np.concatenate(ys).astype(np.float32)
        np.savez(out, pred=preds, y=ys,
                 anchor_time=tr["anchor_time"][np.asarray(va_idx)])
        print("saved", out, len(preds))


if __name__ == "__main__":
    main()
