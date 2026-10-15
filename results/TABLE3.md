# Table 3（范文式）不同天气条件与预测尺度下的框架对比（官方 test）

主结果 = 3-seed 集成；单位 kW。粗体 = 该列最优。
官方 test 为 10 晴 + 10 阴（Nie et al. 2023 固定日期，无雨天）→
Rainy 场景见 WEATHER_EXT.md（val 天扩展口径，晴/阴/雨三分类）。


## 预测尺度：5 min

| Method | MAE sunny | MAE cloudy | MAE overall | RMSE sunny | RMSE cloudy | RMSE overall | R2 sunny | R2 cloudy | R2 overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PGF-TPA (Proposed) | 0.1288 | **2.0237** | **1.0647** | 0.1796 | **3.2452** | **2.2843** | 0.9994 | **0.8048** | **0.9056** |
| Sky-BiMamba (paper) | 0.6030 | 2.1105 | 1.3476 | 0.7129 | 3.3741 | 2.4249 | 0.9898 | 0.7890 | 0.8936 |
| SUNSET | 0.7223 | 2.3639 | 1.5332 | 0.8275 | 3.5701 | 2.5772 | 0.9863 | 0.7638 | 0.8798 |
| ResLSTM | 0.1607 | 2.0688 | 1.1032 | 0.2205 | 3.3240 | 2.3413 | 0.9990 | 0.7952 | 0.9008 |
| ConvLSTM | 0.3081 | 2.1418 | 1.2138 | 0.4074 | 3.3621 | 2.3806 | 0.9967 | 0.7905 | 0.8974 |
| Transformer | 0.1797 | 2.1667 | 1.1611 | 0.2392 | 3.5365 | 2.4913 | 0.9989 | 0.7682 | 0.8877 |
| PatchTST | 0.1325 | 2.2034 | 1.1554 | 0.1784 | 3.6755 | 2.5863 | 0.9994 | 0.7496 | 0.8790 |
| DLinear | 0.4714 | 2.1762 | 1.3135 | 0.5548 | 3.6465 | 2.5930 | 0.9939 | 0.7535 | 0.8783 |
| CrossGNN | **0.1039** | 2.1198 | 1.0996 | **0.1525** | 3.5591 | 2.5037 | **0.9995** | 0.7652 | 0.8866 |

## 预测尺度：15 min

| Method | MAE sunny | MAE cloudy | MAE overall | RMSE sunny | RMSE cloudy | RMSE overall | R2 sunny | R2 cloudy | R2 overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PGF-TPA (Proposed) | **0.2037** | **2.7240** | **1.4397** | **0.2636** | **4.0235** | **2.8240** | **0.9986** | **0.6951** | **0.8524** |
| Sky-BiMamba (paper) | 0.3559 | 2.7952 | 1.5522 | 0.4133 | 4.1337 | 2.9098 | 0.9965 | 0.6782 | 0.8433 |
| SUNSET | 0.3623 | 2.9122 | 1.6129 | 0.4488 | 4.2199 | 2.9725 | 0.9958 | 0.6646 | 0.8364 |
| ResLSTM | 0.2230 | 2.7988 | 1.4863 | 0.3073 | 4.1777 | 2.9339 | 0.9981 | 0.6713 | 0.8407 |
| ConvLSTM | 0.4958 | 2.8328 | 1.6419 | 0.6396 | 4.1433 | 2.9373 | 0.9916 | 0.6767 | 0.8403 |
| Transformer | 0.2280 | 2.8592 | 1.5184 | 0.2984 | 4.2710 | 2.9986 | 0.9982 | 0.6564 | 0.8336 |
| PatchTST | 0.2973 | 3.1109 | 1.6771 | 0.3918 | 4.6627 | 3.2773 | 0.9968 | 0.5905 | 0.8012 |
| DLinear | 1.1009 | 3.1717 | 2.1165 | 1.2874 | 4.7580 | 3.4565 | 0.9658 | 0.5736 | 0.7788 |
| CrossGNN | 0.2195 | 2.9874 | 1.5769 | 0.3412 | 4.5319 | 3.1830 | 0.9976 | 0.6132 | 0.8125 |

## 预测尺度：30 min

| Method | MAE sunny | MAE cloudy | MAE overall | RMSE sunny | RMSE cloudy | RMSE overall | R2 sunny | R2 cloudy | R2 overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PGF-TPA (Proposed) | 0.2928 | **3.1523** | **1.6782** | 0.4231 | **4.5283** | **3.1666** | 0.9961 | **0.6078** | **0.8102** |
| Sky-BiMamba (paper) | 0.3457 | 3.1761 | 1.7171 | 0.4470 | 4.5511 | 3.1841 | 0.9957 | 0.6038 | 0.8081 |
| SUNSET | 0.4931 | 3.2513 | 1.8295 | 0.6197 | 4.6414 | 3.2612 | 0.9917 | 0.5879 | 0.7987 |
| ResLSTM | **0.2805** | 3.1899 | 1.6901 | **0.3937** | 4.6471 | 3.2470 | **0.9967** | 0.5869 | 0.8004 |
| ConvLSTM | 0.5319 | 3.2878 | 1.8672 | 0.7093 | 4.6344 | 3.2658 | 0.9891 | 0.5892 | 0.7981 |
| Transformer | 0.3178 | 3.2196 | 1.7237 | 0.4254 | 4.7279 | 3.3051 | 0.9961 | 0.5724 | 0.7932 |
| PatchTST | 0.5732 | 3.7292 | 2.1024 | 0.7472 | 5.2909 | 3.7217 | 0.9880 | 0.4645 | 0.7378 |
| DLinear | 1.9337 | 3.9528 | 2.9120 | 2.2444 | 5.5580 | 4.1910 | 0.8913 | 0.4091 | 0.6676 |
| CrossGNN | 0.3359 | 3.5773 | 1.9064 | 0.4969 | 5.1281 | 3.5873 | 0.9947 | 0.4970 | 0.7564 |

## Forecast Skill（FS = 1 − RMSE_model / RMSE_RAW-persistence，同尺度）

- 5 min：RAW persistence overall RMSE = 2.7915 kW
  - PGF-TPA (Proposed): FS = +18.2%
  - Sky-BiMamba (paper): FS = +13.1%
  - SUNSET: FS = +7.7%
  - ResLSTM: FS = +16.1%
  - ConvLSTM: FS = +14.7%
  - Transformer: FS = +10.8%
  - PatchTST: FS = +7.4%
  - DLinear: FS = +7.1%
  - CrossGNN: FS = +10.3%
- 15 min：RAW persistence overall RMSE = 3.7073 kW
  - PGF-TPA (Proposed): FS = +23.8%
  - Sky-BiMamba (paper): FS = +21.5%
  - SUNSET: FS = +19.8%
  - ResLSTM: FS = +20.9%
  - ConvLSTM: FS = +20.8%
  - Transformer: FS = +19.1%
  - PatchTST: FS = +11.6%
  - DLinear: FS = +6.8%
  - CrossGNN: FS = +14.1%
- 30 min：RAW persistence overall RMSE = 4.4020 kW
  - PGF-TPA (Proposed): FS = +28.1%
  - Sky-BiMamba (paper): FS = +27.7%
  - SUNSET: FS = +25.9%
  - ResLSTM: FS = +26.2%
  - ConvLSTM: FS = +25.8%
  - Transformer: FS = +24.9%
  - PatchTST: FS = +15.5%
  - DLinear: FS = +4.8%
  - CrossGNN: FS = +18.5%

## Persistence 参照（不入主表竞争）

- RAW persistence @5min: overall RMSE 2.7915 / MAE 1.2395 / R² 0.8590 (sunny 0.3899, cloudy 3.9524)
- CSI persistence @5min: overall RMSE 3.8078 / MAE 2.1702 / R² 0.7376 (sunny 1.2096, cloudy 5.2779)
- RAW persistence @15min: overall RMSE 3.7073 / MAE 2.1139 / R² 0.7456 (sunny 1.1482, cloudy 5.1628)
- CSI persistence @15min: overall RMSE 4.0710 / MAE 2.4098 / R² 0.6932 (sunny 1.4825, cloudy 5.6134)
- RAW persistence @30min: overall RMSE 4.4020 / MAE 3.0197 / R² 0.6332 (sunny 2.2998, cloudy 5.8623)
- CSI persistence @30min: overall RMSE 4.3315 / MAE 2.7251 / R² 0.6449 (sunny 1.9801, cloudy 5.8781)

## 显著性（主方法集成 vs 最强 baseline 集成，平方误差配对 t，双侧）

- sunny@5: vs ResLSTM  t=11.62  p=0.00e+00  n=6126
- cloudy@5: vs ResLSTM  t=4.38  p=1.18e-05  n=5979
- overall@5: vs ResLSTM  t=4.52  p=6.20e-06  n=12105
- sunny@15: vs Sky-BiMamba (paper)  t=49.85  p=0.00e+00  n=5697
- cloudy@15: vs Sky-BiMamba (paper)  t=4.5  p=6.85e-06  n=5483
- overall@15: vs Sky-BiMamba (paper)  t=5.02  p=5.12e-07  n=11180
- sunny@30: vs Sky-BiMamba (paper)  t=3.44  p=5.88e-04  n=5059
- cloudy@30: vs Sky-BiMamba (paper)  t=0.9  p=3.71e-01  n=4755
- overall@30: vs Sky-BiMamba (paper)  t=0.99  p=3.22e-01  n=9814
