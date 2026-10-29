# 历史协议 v1.2：5 分钟尺度扩展（2026-10-12）

本记录保留当时的实验范围。当前可执行训练配方以 `docs/REPRODUCIBILITY.md` 与 `scripts/train.py` 为准。

## 数据和样本

在 v1.1 的 15 分钟预测之外增加 5 分钟预测。输入仍为锚点 t 之前含 t 的 15 帧一分钟天空图像与功率，目标改为 t+5。仅保留输入严格连续且目标存在的样本，索引保存为 `data/proc/{split}_samples_h5.npz`。

日块划分使用 seed=0 的 80/20 训练/验证比例。官方测试集 20 日为 10 晴、10 阴，天气分组为 sunny/cloudy/overall；不增加独立雨天测试日。

## 模型和历史训练规则

基线为 SUNSET、ResLSTM、ConvLSTM、Transformer、PatchTST、DLinear、CrossGNN。各实现适配相同输入和功率单位，架构差异保留在源码与复现说明中。

当时记录的基线规则为 Adam 3e-4、ReduceLROnPlateau（系数 0.5、patience 3）、MSE、batch 256、早停 patience 8、最多 60 轮、种子 0/1/2。TPA 配方的实际轮数与批大小不同，见当前复现说明。

PGF-TPA-RW 使用功率 GRU 与时间池化图像编码器，并将 `abs(y-PV(t))>3 kW` 样本的训练损失权重设为 4：

```bash
python src/run_pgf.py --seed 0 --no-csi --tpa --ramp-w 4 --horizon 5
```

## 结果约定

单次结果为 `results/{model}[_h5]_s{seed}.json` 与对应预测数组。三种子等权集成，单模型均值与样本标准差分开记录。

指标为 sunny/cloudy/overall 的 RMSE、MAE、R² 和 FS；FS 分母为同尺度的原始功率持续性基线。结果表保存在 `results/TABLE3.md` 与 `TABLE3.csv`。历史逐样本配对检验未处理同日相关性，只作为描述性记录。
