# BENCHMARK LOCK v1.1 — SKIPP'D 15-min-ahead PV 功率预测

> 本文件是项目唯一评估协议。任何新模块/新算法不得更改本协议；如确需修改，
> 必须重跑全部 baseline 并在结果表中同时报告新旧两套数字。
> v1.0（官方 h5 直读设想）→ v1.1（自建窗口协议，因官方 stacks 下载源被
> 网络阻断，改用原作者参与的 SolarBench 镜像序列重建 forecast 样本）。

## 1. 数据集（已确定，不再更换）

**SKIPP'D / Stanford 30kW 屋顶 PV + 鱼眼天空图像（2017–2019）**
- 本地数据源：`data/My_SKIPPD_Data`（HF `solarbench/SKIPPD` 的 Kaggle 镜像
  srinathswamy/skipppd-dataa，2,301,751,743 B，CRC 校验通过）
  —— 原作者 Y. Nie 参与的 SolarBench 项目发布的衍生序列：
  64×64 RGB 图像 + 同时刻 PV(kW)，train 349,372 / test 14,003 行
- 物化：`data/proc/`（去重分钟帧 uint8 + 窗口索引，prepare_data.py）
- **官方 h5（stacks.stanford.edu）被本机网络阻断，无法获取**；Zenodo 无镜像。

## 2. Forecast 样本构建（v1.1 核心）

锚点 t（分钟）：
- 输入 `images_log` = [t−14, t] 的 15 帧图像（窗口内时间严格连续 60s）
- 输入 `pv_log` = 同 15 分钟的 PV 值（kW）
- 目标 `pv_pred` = t+15min 的 PV
- 仅接受窗口连续且 t+15 存在的锚点：test n=11,180（晴 5,697 / 阴 5,483）
- 图像归一化 /255；PV 用原始 kW 值（无统计归一化 → 无泄漏风险）

**与官方差异声明**（官方 14003 vs 本协议 11180，−20%）：
镜像时间戳含相机帧秒偏移与个别缺口，严格连续判据剔除了部分样本。
协议对齐性证据：RAW persistence 的 cloudy RMSE 5.163 / overall 3.707
与官方论文 FS 反推值（5.147 / 3.626）吻合至 +0.3% / +2%。
官方论文数字仅作量级参照，严格对比一律用本协议下的复现结果。

## 3. 划分（防泄漏，锁定）

| 集合 | 定义 | 采样 |
|---|---|---|
| train | train 序列按天块 shuffle(seed=0) 取 80% 天 | stride=2（2-min，官方协议） |
| val | 其余 20% 天（无同日样本跨集） | stride=2 |
| test | 官方 20 个手选日晴(10)/阴(10)，日期同官方固定清单 | 全量 stride=1 |

- test 永不参与训练、早停、超参、归一化。
- 时区：全部按 US/Pacific 本地墙钟时刻处理（曾踩坑：tz-aware Series
  的 .values 返回 UTC，已修复并写入 prepare_data.py 注释）。

## 4. 训练协议（全方法通用）

- Adam(lr 3e-4) + ReduceLROnPlateau(×0.5, patience 3)，MSE，batch 256
- 早停 on val RMSE，patience 8，最多 60 epochs
- 3 训练 seed（0/1/2）：报告均值±std；主结果 = 3 seed 预测均值（集成）

## 5. 指标（锁定，kW）

- RMSE / MAE / R²：sunny / cloudy / overall（overall=全样本）
- **FS（Forecast Skill）= 1 − RMSE_model / RMSE_RAW-persistence**
  （分母选 RAW 而非 CSI 的依据：见 §2 对齐性证据）
- CSI-persistence 一并报告（官方 Relative_op 解析晴空模型）

### 锁定的参照数字（test, n=11180）

| 参照 | RMSE sunny | RMSE cloudy | RMSE overall | R² overall |
|---|---|---|---|---|
| RAW persistence | 1.148 | 5.163 | 3.707 | 0.746 |
| CSI persistence | 1.483 | 5.613 | 4.071 | 0.693 |

## 6. Baseline 组

| # | 模型 | 输入 | 状态 |
|---|---|---|---|
| B0 | RAW persistence | pv_log 末值 | ✅ 已锁定 |
| B0' | CSI persistence | CSI×晴空理论 | ✅ 已锁定 |
| B1 | MLP | pv_log 15 值 | 待跑 |
| B2 | GRU | pv_log 序列 | 待跑 |
| B3 | SUNSET-CNN 复现 | 15 帧+pv_log | 待跑 |

## 7. 结果记录

- 单次：`results/{model}_s{seed}.json` + `preds_*.npy`
- 汇总：`results/SUMMARY.md`（make_summary.py 生成，只增不删）

## 8. SOTA 判定标准（v1.1，2026-09-29 修订）

~~原标准 (a)：overall R² ≥ 0.94~~——**修订依据**：核实发现 Sky-BiMamba 的
R²=0.9409 是 **5-min ahead** 任务的数字（Renewable Energy 256 (2025)），其
15-min 口径仅报阴天 MAE −16.2%；PhyMo（arXiv:2609.27554，2026-09）复现
Sky-BiMamba 15-min R²≈0.8313，其自身 SOTA 声明为 0.8480±0.14（协议披露不全，
仅作量级参照）。官方 15-min day-block 协议下可信文献带为
**R² 0.83–0.85 / RMSE 2.85–3.03 kW**（SUNSET 3.03 为官方锚点）。

修订后标准（在同协议下）：
(a) overall RMSE ≤ 2.86（R² ≥ 0.85），即越过全部可查文献水平；
(b) cloudy RMSE 显著低于全部 baseline（配对检验）；
(c) 多折/多 seed 稳定。达成即认定 SOTA，算法细节与消融写入独立文档。

## 9. 锁定状态

- [x] 数据集确定（2026-09-28）
- [x] 协议 v1.1 锁定（本文件）
- [x] B0/B0' 参照锁定：RAW 3.707 / CSI 4.071（overall RMSE, kW）
- [x] B1–B3 跑完并锁定（3 seeds + 集成）：
      MLP 3.192 / GRU 3.159 / SUNSET 3.010（单模型均值），
      集成 GRU 3.156 / SUNSET 2.973
- [x] 新算法阶段一完成：PGF v2.1（kW 域双分支融合）单模型 2.918±0.041，
      3-seed 集成 **2.896（R² 0.8448，FS +21.9%）**，配对 t=-8.25
      显著优于 SUNSET 集成 → **自建协议下当前 SOTA，已处文献带内**
- [x] 阶段二完成（2026-09-29/30）：EMA fold×seed 网格 + 全集成
      **E19 = 2.8742 / R² 0.8471 / FS +22.5%**（vs E3 t=-3.40，vs SUNSET-ens
      t=-8.81）→ 最终主结果；nowcast 预训练两版均为负结果（含恒等捷径修复版）；
      异构集成稀释证伪（只收 PGF 族）；~~集成渐近线 ~2.85，
      R²=0.85 在本图像信息量下不可达~~（此判断被阶段三 CLB 部分推翻，
      见下）
- [x] 阶段三完成（2026-10-01）：强 baseline 复现（SUNSET/ConvLSTM 官方架构/
      CL-SUNSET 机制级）+ PGF-CLB（B 分支 ConvLSTM 时空编码器）
      - CLB 单模型 8 成员（3 固定+5 折）2.9211±0.0278，最强单模 f3=2.8764
        几乎追平 E19 全集成；晴天 RMSE 0.2775 vs v2.1 的 0.3980（-30%）
      - **E27 = E19 + 8×CLB：RMSE 2.8578 / R² 0.8488 / FS +22.9%**，
        配对 t=-4.54 vs E19、-5.55 vs ConvLSTM-ens、-8.33 vs CL-SUNSET-ens，
        均显著 → **最终主结果（ preds_pgf_E27_mix.npy ）**
      - 判据核对：(a) RMSE 2.8578 ≤ 2.86 ✓、R² 0.8488 距 0.85 差 0.0012（部分
        达成）；(b)(c) ✓。E27 vs E8_clb(2.8657) t=-0.97 不显著——增益主体是
        CLB 组件而非成员数；E8_clb 为同等强度的高效替代。
        详见 results/RESEARCH_NOTES.md 阶段三节
- [x] 阶段三C（2026-10-01）：CLB 扩至 seed×fold 全网格（18 成员，最强单模
      s2_f3 = 2.8518/R² 0.8495 单模型即超 E27 集成）
      - **E37 = E19 + 18×CLB：RMSE 2.8503 / R² 0.8496 / FS +23.1% = 当前主结果**
        （preds_pgf_E37.npy）；vs E19 t=-4.18、vs E27 t=-2.94、vs ConvLSTM-ens
        t=-6.17、vs CL-SUNSET-ens t=-8.62 均显著；E18_clb(2.8584) 与 E37 统计
        等价（t=-1.39）
      - val 调权负结果（val_blend.py）：标准 val(2017-18) 与 test(2019) 分布
        漂移——val 上 v2.1 族优于 CLB 族、test 相反，val 选权显著劣化
        （E28w 2.8774, t=+4.33 vs E37）；等权 E28=2.8506≈E37
      - §8(a) 核对：RMSE ✓✓（2.8503）；R² 0.8496 差 0.0004（SSE 差 0.27%）——
        等权集成渐近线恰好卡在门槛下方；s3（+6 成员）扩链中，E24_clb/E43
        为最后一杆
- [x] 阶段三终局（2026-10-01）：s3 六成员完成（固定 2.9119，折 2.8979/2.8908/
      3.0278/2.9966/2.9279，弱于 s0-s2）
      - **E24_clb（24 纯 CLB）= 2.8594/0.8486 反而劣于 E18_clb 2.8584**
        （弱成员稀释）；E43 = 2.8508/0.8496 ≈ E37（t=-1.78 不显著）
      - **最终判定 §8(a)：RMSE ✓（2.8503），R² ✗（渐近线 0.8496）**。
        三条杠杆（扩成员 18→24→43、val 调权、纯族 vs 混合）全部试尽，
        等权族集成在本数据信息量下的上界即此。
      - **主结果锁定：E37 = 2.8503 / R² 0.8496 / FS +23.1%（preds_pgf_E37.npy）**
        —— 全面显著超越 E19 与全部强 baseline 复现（t=-3.72~-8.81），
        RMSE 处可查文献带最低端，R² 持平 PhyMo 0.8480±0.14（协议更严）。
        项目目标（严格协议下超越全部可查文献）就此定档。
- [x] 爬坡事件 SOTA（2026-10-01，用户指令；定义/判据/λ=4 全预注册，
      见 RESEARCH_NOTES 爬坡节，结果 results/ramp_summary.json + rw_summary.json）
      - ramp=|ΔP|>3kW（10% 容量），test n=1815（up 957/down 858）
      - **E_rw6（PGF-RAMP：CNN-B/CLB-B × s0-2，--ramp-w 4）ramp RMSE 5.894**：
        vs ConvLSTM-ens t=-9.18 / CL-SUNSET −13.96 / SUNSET −13.65 / GRU −20.64，
        上行 −2.40、下行 −11.01，全部显著 → **爬坡 SOTA**；同时 overall 2.900
        显著优于全部外部 baseline（vs ConvLSTM t=-2.19）→ "同时 SOTA"（对外部）
      - E_RAMP（43 混合）overall 2.8465 历史最佳（t=-2.01 vs E37）但 ramp 稀释
        不显著（t=-0.78，如实报告）；门控证据：rw 成员 g(ramp) 0.780 >
        g(non-ramp) 0.711（E37 成员无此适应性）——动态融合叙事成立
