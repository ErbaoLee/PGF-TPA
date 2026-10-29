"""Audit the locked TPA evidence and explain ramp weighting from saved predictions.

No fitting, configuration selection or new data. The cost-weighted test risk is
a post-hoc descriptive metric using the existing training weight, not a measured
operating cost. All uncertainty remains conditional on the selected 20 days.
"""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def main():
    locked = json.loads((RES / "tpa_final_analysis.json").read_text(encoding="utf-8"))
    for relative, digest in locked["sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative
    meta = np.load(ROOT / "data/proc/test_meta.npz")
    samples = np.load(ROOT / "data/proc/test_samples.npz")
    y = meta["pv"][samples["tgt"]].astype(float)
    p0 = meta["pv"][samples["win"][:, -1]].astype(float)
    dates = samples["anchor_time"].astype("datetime64[D]").astype(str)
    assert np.all(meta["minutes"][samples["tgt"]].astype("datetime64[s]") ==
                  samples["anchor_time"].astype("datetime64[s]") + np.timedelta64(15, "m"))
    assert np.all(np.diff(meta["minutes"][samples["win"]].astype("datetime64[s]"), axis=1) ==
                  np.timedelta64(60, "s"))
    ramp = abs(y - p0) > 3
    masks = {"overall": np.ones(len(y), bool), "ramp10": ramp, "nonramp": ~ramp,
             "up10": y - p0 > 3, "down10": y - p0 < -3}
    assert len(y) == 11180 and ramp.sum() == 1815
    preds, seeds = {}, []
    for family, base in [("TPA-E3", "pgf_nocsi_tpa"), ("TPA-RW-E3", "pgf_nocsi_tpa_rw4")]:
        members = []
        for seed in range(3):
            p = np.load(RES / f"preds_{base}_s{seed}.npy").astype(float)
            assert p.shape == y.shape and np.isfinite(p).all()
            members.append(p)
            row = {"family": family, "seed": seed}
            for subset, mask in masks.items():
                row[f"{subset}_rmse"] = float(np.sqrt(np.mean((p[mask] - y[mask]) ** 2)))
            seeds.append(row)
        preds[family] = np.mean(members, axis=0)
        for subset, mask in masks.items():
            actual = np.sqrt(np.mean((preds[family][mask] - y[mask]) ** 2))
            assert abs(actual - locked["metrics"][family][subset]["rmse"]) < 1e-12
    q = float(ramp.mean())
    weight = 1 + 3 * ramp
    rows = {}
    for family, p in preds.items():
        err = p - y
        mse = float(np.mean(err ** 2))
        rmse = float(np.sqrt(mse))
        ramp_mse, nonramp_mse = [float(np.mean(err[mask] ** 2)) for mask in [ramp, ~ramp]]
        assert abs(mse - q * ramp_mse - (1 - q) * nonramp_mse) < 1e-12
        rows[family] = dict(overall_rmse=rmse, ramp_rmse=float(np.sqrt(ramp_mse)),
                            nonramp_rmse=float(np.sqrt(nonramp_mse)), overall_mse=mse,
                            ramp_mse_component=q * ramp_mse,
                            nonramp_mse_component=(1 - q) * nonramp_mse,
                            ramp_sse_share=q * ramp_mse / mse,
                            weighted_mse4=float(np.mean(weight * err ** 2)),
                            up_bias=float(np.mean(err[masks["up10"]])),
                            down_bias=float(np.mean(err[masks["down10"]])))
    plain, rw = [rows[n] for n in ["TPA-E3", "TPA-RW-E3"]]
    days = np.unique(dates)
    # Same strata and random seed as the locked whole-day analysis.
    SUNNY = {'2017-09-15', '2017-10-06', '2017-10-22', '2018-02-16', '2018-06-12',
             '2018-06-23', '2019-01-25', '2019-06-23', '2019-07-14', '2019-10-14'}
    rng = np.random.default_rng(20261012)
    strata = [np.flatnonzero(np.isin(days, list(SUNNY))),
              np.flatnonzero(~np.isin(days, list(SUNNY)))]
    draws = np.concatenate([rng.choice(s, (20000, len(s))) for s in strata], axis=1)
    counts = np.array([np.sum(dates == d) for d in days])
    den = counts[draws].sum(axis=1)
    boot = {}
    for name, p in preds.items():
        sums = np.array([np.sum((weight * (p - y) ** 2)[dates == d]) for d in days])
        boot[name] = sums[draws].sum(axis=1) / den
    ci = np.quantile(boot["TPA-RW-E3"] - boot["TPA-E3"], [.025, .975]).tolist()
    out = dict(protocol=locked["protocol"], date="2026-10-12",
               audited_prediction_files=len(locked["sha256"]), ramp_fraction=q,
               models=rows, singles=seeds,
               weighted_mse4_delta=rw["weighted_mse4"] - plain["weighted_mse4"],
               weighted_mse4_relative_reduction=1 - rw["weighted_mse4"] / plain["weighted_mse4"],
               weighted_mse4_day_bootstrap_95ci=ci,
               inference_scope="Post-hoc, unadjusted, conditional on the selected 20 test days",
               validation_selection="Unweighted overall validation RMSE for both losses; see run_pgf.py")
    (RES / "tpa_loss_utility.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    with (RES / "tpa_loss_seed_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(seeds[0])); writer.writeheader(); writer.writerows(seeds)
    lines = ["# TPA 爬坡加权损失指标分析", "",
             "由已保存预测复算，保持协议、测试集合和配置不变。", "",
             "## 直接结论", "",
             "爬坡加权损失有用，其作用是降低发生较大净功率变化时的功率预测误差，代价是平稳段误差增加。它没有单独预测爬坡标签，也尚未验证告警命中率、提前量或实际调度收益。", "",
             "## 同规模对照", "",
             "|配置|总体RMSE/kW|爬坡RMSE/kW|非爬坡RMSE/kW|上行偏差/kW|下行偏差/kW|",
             "|---|---:|---:|---:|---:|---:|"]
    for name, row in rows.items():
        lines.append(f"|{name}|{row['overall_rmse']:.4f}|{row['ramp_rmse']:.4f}|{row['nonramp_rmse']:.4f}|{row['up_bias']:+.4f}|{row['down_bias']:+.4f}|")
    lines += ["", "偏差定义为预测减真实。加权后上行低估和下行高估均有所减轻，但仍存在；该现象不能单独证明已消除时间滞后。", "",
              "## 三种子的一致性", "",
              "|种子|普通总体|加权总体|普通爬坡|加权爬坡|", "|---|---:|---:|---:|---:|"]
    for seed in range(3):
        a, b = [next(r for r in seeds if r["seed"] == seed and r["family"] == family) for family in ["TPA-E3", "TPA-RW-E3"]]
        lines.append(f"|{seed}|{a['overall_rmse']:.4f}|{b['overall_rmse']:.4f}|{a['ramp10_rmse']:.4f}|{b['ramp10_rmse']:.4f}|")
    lines += ["", "三个种子均降低爬坡RMSE、提高总体RMSE，因此观察不只来自集成平均。种子共享数据与验证策略，并非三次独立外部验证。", "",
              "## 误差构成", "",
              f"爬坡窗口占 {100*q:.2f}%；普通TPA中贡献 {100*plain['ramp_sse_share']:.2f}% 的总平方误差，加权后为 {100*rw['ramp_sse_share']:.2f}%。",
              "RMSE不可以线性相加，正确分解是 MSE总体 = q×MSE爬坡 + (1−q)×MSE非爬坡。", "",
              f"爬坡项变化 {rw['ramp_mse_component']-plain['ramp_mse_component']:+.4f} kW²；非爬坡项变化 {rw['nonramp_mse_component']-plain['nonramp_mse_component']:+.4f} kW²；合计 {rw['overall_mse']-plain['overall_mse']:+.4f} kW²。",
              "这直接解释了爬坡误差降低而总体RMSE上升，无需假设矛盾或训练失败。", "",
              "## 与训练目标对应的补充指标", "",
              "定义 J4 = mean[(1+3×真实爬坡指示)×平方误差]，单位kW²，与当前训练损失相同，分母为全部窗口数。它是事后描述指标，不是新增主评估指标，也不是实际经济损失。", "",
              f"普通TPA J4={plain['weighted_mse4']:.4f}，加权TPA J4={rw['weighted_mse4']:.4f}，下降 {100*out['weighted_mse4_relative_reduction']:.2f}%。",
              f"加权减普通差值的晴阴分层整日配对95%区间 [{ci[0]:+.4f}, {ci[1]:+.4f}] kW²；使用20,000次抽样，条件于这20日，未校正多重比较。", "",
              "## 训练与评估的准确表述", "",
              "run_pgf.py 的训练目标使用rw4，但学习率调整、早停及raw/EMA选择仍依据未加权的总体验证RMSE。因此该实验验证的是加权训练在现有总体选模策略下的效果，不是专门按爬坡验证误差选出的事件最优模型。", "",
              f"审计：核对锁定分析记录的 {len(locked['sha256'])} 个预测文件SHA256、窗口连续性、目标时距以及两组集成的五项子集指标。数据依据：tpa_final_analysis.json、tpa_loss_utility.json；复算：src/analyze_tpa_loss_utility.py。"]
    (RES / "TPA_LOSS_UTILITY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
