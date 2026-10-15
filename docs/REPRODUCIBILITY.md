# 复现路线与文件约定

## 三种复现路线

1. **已存预测复算**：安装 `requirements-metrics.txt`，运行 `scripts/reproduce.py`。不读取图像或训练模型；固定数据与预测直接重算结果。
2. **已存权重推理**：下载/重建图像后运行 `src/infer.py --check-stored`。读取真实state_dict并检查预测，输出新文件至 `outputs/inference/`。
3. **重新训练**：运行 `scripts/train.py`，产物至 `outputs/retrained/`。不同硬件/版本/非确定性训练可能产生多种子统计范围内的差别。

## 依赖

建议Python3.12。主流程依赖NumPy、pandas、PyArrow、Pillow、requests、PyTorch与时区数据；参考结果复算只依赖NumPy。没有强制依赖CUDA、torchvision或mamba-ssm，Sky-BiMamba保留纯PyTorch扫描后备实现。旧探索性诊断脚本 `diagnose_protocol.py` 需要额外安装Hugging Face datasets；不属于README主复现路径。

本次实际检查的环境与结果记录在 [VALIDATION.md](VALIDATION.md)。`requirements.txt` 为兼容范围，不等价于历史GPU环境锁定。未保留下来的原始训练环境不能据此补造。

## 下载与数据一致性

`scripts/download_data.py` 读取原始dataset_info记录的六个Parquet文件名与字节数，从固定提交下载。每个文件先写 `.partial`，检查HTTP状态、长度及Parquet结构后再更名。上游网络或服务不可用时，可使用已有Arrow目录，不会自动改用当前最新版。

`src/prepare_data.py` 支持Parquet和Hugging Face Arrow。重建时会生成三套索引。`data/manifest.json` 保存各NPZ内部数组的形状、dtype和内容SHA-256，以及完整图像NPY文件的SHA-256。复现检查不会要求NPZ的容器时间戳一致：

```bash
python scripts/verify_assets.py --skip-assets --arrays --images
```

`src/prep_clearsky.py` 保留原参考函数和异常时nanmedian填补规则，以保持权重输入一致。缓存已包含在仓库；重建后可检查数组内容。解析晴空三角函数在不同数值环境下有末位差异：本次最大约8.5e-15 kW，因此该数组与独立冻结参考按绝对容差1e-10 kW核对，其余数据数组仍要求SHA-256完全一致。PNG解码/Pillow变化若导致图像散列不符，先检查环境版本，不能直接跳过校验后宣称精确复现。

## 权重、结果与缺失项

所有可用的项目权重均以实际二进制文件保留，见 `checkpoints/index.json`。权重仍放在 `results/*_ckpt.pt` 以兼容原训练/评价代码；另有两个预训练辅助权重位于 `data/proc/`。全部二进制资产由Git LFS追踪，`checkpoints/assets.json` 记录大小及SHA-256。

文件基本格式：`pgf_nocsi_tpa[_rw4][_h5|_h30]_s{seed}[_f{fold}]_ckpt.pt`。15分钟不带horizon后缀。`_res_l10.5` 为历史持久性残差+L1补充配方，不替代普通TPA或TPA-RW。历史AWA、辅助损失、混合集成预测属于探索性资产。

原始MLP/GRU/部分SUNSET、ConvLSTM实验只保存了预测，不能凭空恢复其权重。SUNSET与ConvLSTM的15分钟 `_r1` 权重属于之后的重跑，和旧主表使用的无 `_r1` 预测不是同一训练实例；分别保留并标明。`checkpoints/LOAD_CHECKS.json` 记录当前代码对已有checkpoint的加载检查，历史架构不兼容项不得冒充可精确推理的主模型。

如需直接调用原入口，可以指定输出目录。PowerShell示例：

```powershell
$env:PGF_RESULTS_DIR = "$PWD/outputs/my_run"
python src/run_pgf.py --no-csi --tpa --ramp-w 4 --seed 0 --epochs 40 --batch-size 128
```

读取本地外置图像数据时，可设置 `PGF_DATA_DIR` 指向含 `proc/` 的data目录。默认所有路径相对脚本自身解析，不依赖开发者绝对目录。

## 结果文件的用途

`scripts/reproduce.py` 生成可直接复核的多尺度表。`src/make_table3.py`、`src/eval_weather_ext.py` 保留旧表重算入口，运行时会写入results；正式发布数据核对优先使用输出独立的reproduce入口。`src/analyze_tpa_loss_utility.py` 可检查15分钟普通/加权损失的历史证据。

`results/*.json/.csv/.md` 同时保留阶段性实验记录，不能把全部文件视为单一“最终最优模型”。数值参照见 `checkpoints/reference_metrics.json`，主配置与解释见README和PROTOCOL。
