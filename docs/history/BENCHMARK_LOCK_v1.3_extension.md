# BENCHMARK_LOCK v1.3 —— 论文对标扩展协议（2026-10-12 锁定）

本文件在 v1.1（15 min 主协议）与 v1.2（5 min）之上，锁定与
Renewable Energy 256 (2026) 124361（Sky-BiMamba）全面对标的扩展口径。
v1.1 的所有条款不变；以下为增补。

## 1. 预测尺度扩展（5 / 15 / 30 min）

- 锚点 t，输入 [t−14, t] 15 帧天空图像 + 15 个 PV 值（与 v1.1 完全一致，
  仅目标改为 t+5 / t+15 / t+30 的 PV）。
- 样本索引文件：`data/proc/{split}_samples_h5.npz`（v1.2 已建）、
  `data/proc/{split}_samples_h30.npz`（v1.3 新建：test 9814 / train 220548）。
- 结果命名：15 min 沿用旧名（`{model}_s{seed}`），5/30 min 加 `_h5`/`_h30` 后缀。
- 30 min 为论文表 2/表 3 的最长尺度；5 min 为最短。均需 3 seeds {0,1,2}。

## 2. 天气场景扩展（Sunny / Cloudy / Rainy）

**问题**：SKIPP'D 官方 test 20 日（Nie et al. 2023 固定日期）为 10 晴 + 10 阴，
无雨天 → 论文表 3 的 Rainy 列无法在官方 test 上复现。

**方案**：val 天扩展口径——
1. 规则分类器对全部 497 个训练日分类（`src/classify_weather.py`）：
   - 特征：日晴空指数 clearness = ΣPV / ΣPV_clearsky（理论晴空发电来自
     `{split}_clearsky.npz`）；归一化波动 var_n = mean|diff(晴空指数序列)|。
   - 规则：sunny ⇔ var_n < 0.02 且 clearness ≥ 0.70；rainy ⇔ clearness < 0.45；
     其余 cloudy。**在官方 20 测试日上校准 20/20 全对**（晴/阴两类）。
   - 全数据：320 晴 / 172 阴 / 25 雨。
2. 雨天只从 **val 天**（day-block 20%，训练从未见过的日期）中取：
   6 个雨天（2017-08-04、2017-11-04、2018-01-17、2018-01-18、2018-03-10、2018-05-16）。
   val 天构成：61 晴 / 32 阴 / 6 雨。
3. 各模型训练结束时对 val 天 stride=1 全量预测并保存
   `{stem}_valfull.npz`（pred / y / anchor_time，h15 时 51826 样本）；
   已有 ckpt 用 `src/backfill_valfull.py` 回填。
4. 天气表（`src/eval_weather_ext.py` → `results/WEATHER_EXT.md`）：
   3 seeds 集成 × 5/15/30 min × 晴/阴/雨 × MAE/RMSE/R²，附配对 t 显著性。

**解读限制（必须写进论文文本）**：val 曾用于所有模型的早停（对各模型同等对待），
rainy 列为扩展口径，与官方 test 的晴/阴列不直接同表混排；主表（晴/阴/overall）
仍以官方 test 为准。SUNSET/ConvLSTM 的 h15 旧结果无 valfull，已在 v1.3 队列中
重跑（结果文件 _r1 后缀，评价脚本自动取最新）。

## 3. 爬坡事件口径（论文 §4.1.4 对标）

- 爬坡率 ΔP = PV(t+15) − PV(t)，官方 test，15 min 尺度。
- **主判据 |ΔP| > 2 kW**（论文原文 "exceeded 2 kW/15 min"）；
  敏感性 **|ΔP| > 3 kW**（τ = 0.4 p.u./h 对 30 kW 机组 = 12 kW/h = 3 kW/15 min；
  论文文字与 τ 换算不一致，故双阈值并报）。
- 四分类（论文表 4）：显著上/下爬、非显著上/下爬。
- 指标：爬坡子集上 MAE / RMSE / R² + 上/下行拆分 + 配对 t。
- 脚本：`src/ramp_eval_paper.py` → `results/RAMP_PAPER.md` + `fig_ramp_events.png`。

## 4. 对标模型清单（9 框架 × 3 尺度 × 3 seeds）

| 方法 | 实现 | 图像 |
|---|---|---|
| PGF-TPA-RW（本文） | `run_pgf.py --no-csi --tpa --ramp-w 4` | ✓ |
| Sky-BiMamba（论文本体复现） | `run_skybimamba.py`（6.25M 参数，双分支 BiMamba×12 + 两阶段融合） | ✓ |
| SUNSET | `run_baselines.py --model sunset`（CNN+LSTM） | ✓ |
| ResLSTM | `run_baselines.py --model reslstm`（ResNet18+LSTM） | ✓ |
| ConvLSTM | `run_baselines.py --model convlstm` | ✓ |
| Transformer | `run_baselines.py --model transformer` | ✗（PV 序列） |
| PatchTST | `run_baselines.py --model patchtst` | ✗ |
| DLinear | `run_baselines.py --model dlinear` | ✗ |
| CrossGNN | `run_baselines.py --model crossgnn` | ✗ |

Sky-BiMamba 复现注记：协议与全部 baseline 完全一致（Adam 3e-4 / 逻辑 batch 256
（物理微批 32 + 梯度累积）/ plateau / 早停 8 / EMA 双轨评估 / 梯度裁剪 1.0），
另加 bf16 训练混精与 48 个 MambaBlock 的 torch.compile（数值策略，不改协议）。

## 5. 结果文件与产物

- `results/TABLE2.md`：论文表 2 式（Framework × MAE/RMSE/R² × 5/15/30，overall）
- `results/TABLE3.md/.csv/_std.md/_sig.json`：论文表 3 式（天气 × 尺度）+ FS + 显著性
- `results/WEATHER_EXT.md/.json`：雨天扩展口径表
- `results/RAMP_PAPER.md` + `fig_ramp_events.png`：爬坡（图 12 式）
- `results/fig_horizon_bars.png`（图 7 式）、`fig_weather_curves.png`（图 8 式）、
  `fig_rainy_curves.png`（图 8 雨天扩展）、`fig_radar.png`（图 9 式 ×3 尺度）、
  `fig_error_dist.png`（图 10 式）

## 6. SOTA 判定标准

官方 test 主表（overall/sunny/cloudy × 5/15/30 × MAE/RMSE/R²）逐 cell：
PGF-TPA-RW 集成须优于其余 8 框架集成；rainy 扩展列同标准。
任何 cell 失守 → 记录并触发主方法迭代（不改动 baseline）。
