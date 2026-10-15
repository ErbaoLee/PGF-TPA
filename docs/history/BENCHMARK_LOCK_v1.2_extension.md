# BENCHMARK LOCK v1.2 扩展（2026-10-12）——新增 5 分钟预测尺度

> 本文件是 v1.1（BENCHMARK_LOCK.md）的**增量扩展**，不修改 v1.1 任何既有规则。
> 15 分钟协议（数据、划分、指标、既有全部结果）原样保留、继续有效。
> 动机：对齐范文（paper/Sky-BiMamba_RenewableEnergy2026_reference.pdf，
> Renewable Energy 256 (2026) 124361，表 2/表 3 为 horizon × 天气 × MAE/RMSE/R²）
> 的多维对比设计：在天气条件（晴/阴）× 预测尺度（5/15 min）两个维度上
> 展开主表。

## 1. 范文对齐说明（baseline 选取与对比场景）

- **范文 baseline 组（7 个）**：SUNSET、ResLSTM、ConvLSTM、Transformer（多模态）
  ＋ PatchTST、DLinear、CrossGNN（单模态时间序列）。
- **本工程对应**：SUNSET（官方架构复现，已有）、ConvLSTM（Cambridge 官方复现，
  已有）＋ 本次新增 ResLSTM、Transformer、PatchTST、DLinear、CrossGNN
  （按范文 §3.3.1 描述复现，适配 15 帧 1-min 输入，偏差记录于 README）。
- **对比场景**：范文的 Sunny/Cloudy/Rainy 三分类在本数据不可全支撑——
  SKIPP'D 官方 20 个 test 日为 10 晴 + 10 阴（Stanford 地中海气候，官方
  协议无雨天 test 日）。故本工程天气维度 = sunny / cloudy / overall（v1.1
  锁定口径），在 README 中如实声明差异。
- **预测尺度**：范文为 5/15/30 min；按任务要求取 **5 min 与 15 min** 两档。

## 2. 5 分钟尺度规则（与 v1.1 完全同构，仅目标位移改变）

- 输入不变：锚点 t，`images_log`/`pv_log` = [t−14, t] 15 帧严格连续 60s。
- 目标：`pv_pred` = **t+5 min** 的 PV。
- 仅接受窗口连续且 t+5 存在的锚点（因此 h5 样本集是 h15 的超集或近似）。
- 物化文件：`data/proc/{split}_samples_h5.npz`（anchors/win/tgt/anchor_time）。
- 划分：day-block shuffle(seed=0) 80/20，与 v1.1 同规则同种子。
- 训练协议：Adam(3e-4) + ReduceLROnPlateau(×0.5, patience 3)，MSE，
  batch 256，早停 patience 8、最多 60 epochs，3 seeds（0/1/2）。
- 指标：RMSE/MAE/R²（sunny/cloudy/overall，kW）+ FS（分母 = h5 的 RAW
  persistence）。爬坡感知加权（ramp-w=4）仅用于主方法，与 15 min 一致。

## 3. 主方法（本工程提出模型）

**PGF-TPA-RW**：双分支物理门控融合（A=GRU kW 域；B=TPA 时间池化图像编码器）
+ 爬坡感知损失加权（|y−PV(t)|>3kW 样本权重×4）。配置：
`python src/run_pgf.py --seed {0,1,2} --no-csi --tpa --ramp-w 4 [--horizon 5]`
- 主结果 = 3-seed 预测均值（集成），单模型均值±std 一并报告。
- 15 min 既有：单模型 2.9196±0.0056（RMSE, overall），爬坡子集 5.83
  （纯 TPA 6.12，rw 的设计动机）。

## 4. 结果命名

- 单次：`results/{model}[_h5]_s{seed}.json` + `preds_...npy`（h5 = 5 min 尺度）
- 主表：`results/TABLE3.md` / `TABLE3.csv`（范文表 3 同构：方法 × 天气 ×
  尺度 × MAE/RMSE/R²，最优加粗，含显著性配对 t 检验与 FS）。

## 5. SOTA 判定（v1.2，在 v1.1 §8 基础上增加 5 min 条款）

- 15 min：沿用 v1.1 §8（对照文献带 RMSE 2.85–3.03 / R² 0.83–0.85）。
- 5 min：无外部文献锚点（官方 SKIPP'd 论文未报 5-min forecast），SOTA 判定
  = 在同协议下显著优于全部 7 个 baseline（配对 t，p<0.05）且 3 seed 稳定。
- 双尺度要求：主方法在 5 与 15 两个尺度上均须优于全部 baseline 方可声称
  "our standard 下 SOTA"（对齐范文表 3 中 Proposed 全表最优的呈现方式）。
