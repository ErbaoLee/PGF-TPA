# SOTA 达标审计（PGF-TPA-RW vs 全部 baseline）

指标格子总数 27：第一 21，次优 4，第三及以后 2

| 维度 | 尺度 | 场景 | 指标 | PGF值 | 排名 | 参赛数 | 最优方法 |
|---|---|---|---|---|---|---|---|
| TABLE3 | 5 | sunny | mae | 0.1288 | 2 **次优** | 9 | CrossGNN |
| TABLE3 | 5 | sunny | rmse | 0.1796 | 3 **FAIL** | 9 | CrossGNN |
| TABLE3 | 5 | sunny | r2 | 0.9994 | 3 **FAIL** | 9 | CrossGNN |
| TABLE3 | 5 | cloudy | mae | 2.0237 | 1 | 9 | PGF-TPA (Proposed) |
| TABLE3 | 5 | cloudy | rmse | 3.2452 | 1 | 9 | PGF-TPA (Proposed) |
| TABLE3 | 5 | cloudy | r2 | 0.8048 | 1 | 9 | PGF-TPA (Proposed) |
| TABLE3 | 5 | overall | mae | 1.0647 | 1 | 9 | PGF-TPA (Proposed) |
| TABLE3 | 5 | overall | rmse | 2.2843 | 1 | 9 | PGF-TPA (Proposed) |
| TABLE3 | 5 | overall | r2 | 0.9056 | 1 | 9 | PGF-TPA (Proposed) |
| TABLE3 | 15 | sunny | mae | 0.2037 | 1 | 9 | PGF-TPA (Proposed) |
| TABLE3 | 15 | sunny | rmse | 0.2636 | 1 | 9 | PGF-TPA (Proposed) |
| TABLE3 | 15 | sunny | r2 | 0.9986 | 1 | 9 | PGF-TPA (Proposed) |
| TABLE3 | 15 | cloudy | mae | 2.7240 | 1 | 9 | PGF-TPA (Proposed) |
| TABLE3 | 15 | cloudy | rmse | 4.0235 | 1 | 9 | PGF-TPA (Proposed) |
| TABLE3 | 15 | cloudy | r2 | 0.6951 | 1 | 9 | PGF-TPA (Proposed) |
| TABLE3 | 15 | overall | mae | 1.4397 | 1 | 9 | PGF-TPA (Proposed) |
| TABLE3 | 15 | overall | rmse | 2.8240 | 1 | 9 | PGF-TPA (Proposed) |
| TABLE3 | 15 | overall | r2 | 0.8524 | 1 | 9 | PGF-TPA (Proposed) |
| TABLE3 | 30 | sunny | mae | 0.2928 | 2 **次优** | 9 | ResLSTM |
| TABLE3 | 30 | sunny | rmse | 0.4231 | 2 **次优** | 9 | ResLSTM |
| TABLE3 | 30 | sunny | r2 | 0.9961 | 2 **次优** | 9 | ResLSTM |
| TABLE3 | 30 | cloudy | mae | 3.1523 | 1 | 9 | PGF-TPA (Proposed) |
| TABLE3 | 30 | cloudy | rmse | 4.5283 | 1 | 9 | PGF-TPA (Proposed) |
| TABLE3 | 30 | cloudy | r2 | 0.6078 | 1 | 9 | PGF-TPA (Proposed) |
| TABLE3 | 30 | overall | mae | 1.6782 | 1 | 9 | PGF-TPA (Proposed) |
| TABLE3 | 30 | overall | rmse | 3.1666 | 1 | 9 | PGF-TPA (Proposed) |
| TABLE3 | 30 | overall | r2 | 0.8102 | 1 | 9 | PGF-TPA (Proposed) |

## 次优明细（容忍 ≤2）
- TABLE3 h5 sunny mae: PGF 0.1288 vs 最优 0.1039
- TABLE3 h30 sunny mae: PGF 0.2928 vs 最优 0.2805
- TABLE3 h30 sunny rmse: PGF 0.4231 vs 最优 0.3937
- TABLE3 h30 sunny r2: PGF 0.9961 vs 最优 0.9967

## FAIL 明细（第三及以后，必须迭代）
- TABLE3 h5 sunny rmse: PGF 0.1796 排名3 vs 最优 0.1525
- TABLE3 h5 sunny r2: PGF 0.9994 排名3 vs 最优 0.9995

## 结论：FAIL
