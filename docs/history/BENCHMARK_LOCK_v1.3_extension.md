# 历史协议 v1.3：多尺度、天气与爬坡扩展（2026-10-12）

本记录保留当时的实验范围，当前数据与运行规则以 `docs/PROTOCOL.md` 和 `docs/REPRODUCIBILITY.md` 为准。

## 预测尺度

输入为锚点 t 之前含 t 的 15 帧天空图像与功率，目标为 t+5、t+15 或 t+30 分钟。15 分钟结果使用 `{model}_s{seed}`，其余增加 `_h5` 或 `_h30`，种子均为 0/1/2。30 分钟样本索引为 `data/proc/{split}_samples_h30.npz`，训练样本 220548、测试样本 9814。

## 天气扩展

官方测试集为 10 晴和 10 阴，没有独立雨天测试日。历史天气扩展对 497 个训练日分类：

- clearness = 日功率总和 / 解析晴空功率总和。
- var_n = 晴空指数相邻变化绝对值均值。
- sunny：var_n < 0.02 且 clearness ≥ 0.70；rainy：clearness < 0.45；其余 cloudy。

全体日期规则标签为 320 晴、172 阴、25 雨；验证日期为 61 晴、32 阴、6 雨。6 个规则标记雨日为 2017-08-04、2017-11-04、2018-01-17、2018-01-18、2018-03-10、2018-05-16。验证集曾用于早停，因此这些标签与结果属于探索性分析，不构成独立雨量观测或独立雨天测试。

`src/classify_weather.py` 生成规则标签，`src/backfill_valfull.py` 回填完整验证预测，`src/eval_weather_ext.py` 汇总 `results/WEATHER_EXT.md/.json`。15 分钟完整验证预测包含 51826 样本，NPZ 保存 pred/y/anchor_time。部分旧基线使用 `_r1` 重跑结果。

## 爬坡评估

定义 ΔP = PV(t+15) − PV(t)。历史扩展同时记录 `abs(ΔP)>2 kW` 与 `abs(ΔP)>3 kW`，包括上行、下行、非显著上行和非显著下行分组。结果为 `results/RAMP_THRESHOLDS.md` 和 `ramp_thresholds.json`；当前主复现入口的爬坡阈值仍为 3 kW，两种口径不能混用。

记录 MAE/RMSE/R² 与历史逐样本配对统计，后者未处理同日相关性。

## 对照模型与结果

包括 PGF-TPA-RW、Sky-BiMamba、SUNSET、ResLSTM、ConvLSTM、Transformer、PatchTST、DLinear 和 CrossGNN。Sky-BiMamba 的历史配方使用逻辑 batch 256、物理微批 32 与梯度累积、bf16、EMA、梯度裁剪 1.0，以及可选 torch.compile；实际训练与实现差异见源码和当前复现说明。

数值文件包括总体多尺度表 `results/TABLE2.md`，天气与尺度表 `TABLE3.md/.csv/_std.md/_sig.json`，天气扩展 `WEATHER_EXT.md/.json` 和双阈值爬坡结果。不同协议之间的数值不能直接比较。
