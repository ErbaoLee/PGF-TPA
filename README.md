# PGF-TPA

**天空图像与光伏功率历史的超短期预测：Physics-Gated Fusion + Temporal-Pooled Aggregation。**

本仓库提供完整的训练、数据重建、权重推理和结果复算流程，覆盖 **5 / 15 / 30 分钟**预测。输入为连续15分钟的64×64 RGB天空图像与功率记录，输出一个未来时刻的光伏功率，单位kW。

English instructions: [README_EN.md](README_EN.md)。数据、模型及协议细节见 [复现说明](docs/REPRODUCIBILITY.md) 与 [实验协议](docs/PROTOCOL.md)。

## 已包含的内容

- PGF、TPA、TPA-RW及消融模型；MLP、GRU、SUNSET、ConvLSTM、ResLSTM、Transformer、PatchTST、DLinear、CrossGNN与Sky-BiMamba适配基线。
- 实际模型权重、逐样本预测、验证预测、样本索引、功率/时间元数据和晴空功率缓存。
- 固定上游版本的数据下载与处理脚本，以及SHA-256校验清单。
- 数据划分、训练入口、原始结果表和可重复计算的指标。

论文绘图、图册、草稿、生成的图片、远程登录/传输脚本、下载碎片与重复图像缓存不纳入仓库。**原始天空图像从上游下载后重建**，不重复提交4.3GB的训练图像缓存。

## 1. 安装与克隆

建议使用Python 3.12及独立虚拟环境。训练建议使用CUDA GPU；结果复算只需要NumPy。

```bash
git lfs install
git clone https://github.com/ErbaoLee/PGF-TPA.git
cd PGF-TPA
git lfs pull
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

如需GPU版PyTorch，请按 [PyTorch官方安装页面](https://pytorch.org/get-started/locally/) 选择与驱动匹配的构建。GitHub网页下载ZIP可能只包含LFS指针；推荐使用上面的Git命令。权重文件名与散列见 [权重索引](checkpoints/README.md)。

## 2. 不下载图像，直接复算结果

```bash
python -m pip install -r requirements-metrics.txt
python scripts/verify_assets.py
python scripts/reproduce.py
```

结果写入 `outputs/reproduction/metrics.csv`、`metrics.json` 和 `RESULTS.md`。脚本检查样本对齐、预测长度与固定参照数值，重新计算晴天、阴天、总体及爬坡子集的RMSE / MAE / R² / FS。

15分钟预测的代表结果如下；E3表示种子0/1/2的预测等权平均，E18为3个固定划分成员加15个交叉验证成员。

| 配置 | 总体RMSE (kW) | 爬坡RMSE (kW) |
|---|---:|---:|
| TPA-E3（普通MSE） | 2.8240 | 6.1168 |
| TPA-RW-E3（爬坡加权） | 2.8799 | 5.8343 |
| TPA-E18（普通MSE） | 2.8125 | 6.2004 |
| ConvLSTM-E3 | 2.9373 | 6.2745 |

爬坡定义为 `abs(P(t+15)-P(t)) > 3 kW`。TPA-RW面向爬坡事件；普通TPA具有更低的总体误差，二者存在取舍。5/15/30分钟普通TPA-E3总体RMSE分别为2.2843 / 2.8240 / 3.1666 kW。完整原始表见 [TABLE3](results/TABLE3.md)。

## 3. 重建实验数据

数据来自 [SolarBench SKIPPD](https://huggingface.co/datasets/solarbench/SKIPPD)。本项目固定使用原实验元数据记录的上游提交 `f51690de73d34cc409670ef6119fd69cc3aade8d`，避免上游更新改变样本。原始数据约2.32GB，物化图像约4.47GB，建议预留至少12GB空闲空间。

```bash
python scripts/download_data.py
python src/prepare_data.py
python src/prep_clearsky.py
python scripts/verify_assets.py --skip-assets --arrays --images
```

若已有原实验的 `My_SKIPPD_Data` Arrow目录，可跳过下载：

```bash
python src/prepare_data.py --source /path/to/My_SKIPPD_Data
python src/prep_clearsky.py
```

处理过程保持US/Pacific当地墙钟时间、原始kW单位与严格60秒连续窗口。`prepare_data.py` 一次生成全部三个预测尺度；校验按数组内容进行，NPZ容器头的差异不会影响数据一致性。详见 [数据说明](data/README.md)。

## 4. 从权重执行预测

准备好图像后，无需重新训练即可运行：

```bash
# 三个TPA-RW成员在完整15分钟测试集上推理并与已存预测核对
python src/infer.py --family tpa-rw --horizon 15 --check-stored
# 普通TPA；5和30分钟同样支持
python src/infer.py --family tpa --horizon 5 --check-stored
# 仅前32个样本，验证安装与加载
python src/infer.py --family tpa-rw --seeds 0 --limit 32 --check-stored
# 直接指定任意兼容权重
python src/infer.py --checkpoint results/pgf_nocsi_tpa_s0_f0_ckpt.pt --check-stored
```

输出位于 `outputs/inference/`。`--limit` 的指标只针对样本前缀，不能作为论文完整测试结果。CPU/GPU和卷积算法之间允许小量浮点差异；检查阈值为 `atol=1e-3, rtol=1e-4` kW。

## 5. 从头训练

```bash
python scripts/train.py --family tpa-rw --horizons 15 --seeds 0 1 2
python scripts/train.py --family tpa --horizons 5 15 30 --seeds 0 1 2
python scripts/train.py --family convlstm --horizons 15 --seeds 0 1 2
python scripts/train.py --family tpa-rw --dry-run
```

新结果保存在 `outputs/retrained/`，保留仓库中的固定结果。TPA配方采用40轮上限、batch128、Adam 3e-4、验证集早停与raw/EMA选择；基线的实际配方见 [复现说明](docs/REPRODUCIBILITY.md)。原实验没有启用完全确定性的CUDA算法，因此从头训练不保证逐位复现；**已有权重推理**与**已存预测复算**是复现发布结果的直接入口。

## 目录

```text
src/                 模型、训练、推理与数据处理
scripts/             固定版本下载、训练配方、指标复算与文件校验
data/proc/           功率、时间、窗口索引、晴空缓存；图像在本地重建
results/             权重、逐样本预测、原始指标与补充实验
checkpoints/         权重索引、散列、资产清单及数值参照
docs/                协议、复现说明、发布验证与历史协议
outputs/             本地重跑输出，Git忽略
```

## 结果解释与许可证

本仓采用自建连续窗口协议，15分钟测试样本11,180个，而镜像源测试行数为14,003；不能与不同协议论文的数字直接比较或据此宣称全面SOTA。官方20个测试日包含10晴/10阴；`WEATHER_EXT`中的“雨天”是功率特征规则标签，来自曾参与早停的验证日，是探索性分析，不是独立雨天测试。历史逐样本显著性统计受同日相关性影响，仅作描述性参考。

代码及本项目训练权重沿用目标仓库已有的 [GPL-3.0许可证](LICENSE)。上游数据遵循其自身条款与署名要求，不能改用代码许可证覆盖。原作者晴空参考函数保留原署名；详见 [第三方说明](THIRD_PARTY_NOTICES.md)。本仓未提供尚未确认的论文DOI、作者列表或虚构引用。
