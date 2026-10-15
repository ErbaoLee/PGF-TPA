# 数据来源与重建

上游：[solarbench/SKIPPD](https://huggingface.co/datasets/solarbench/SKIPPD)。固定提交：`f51690de73d34cc409670ef6119fd69cc3aade8d`。`upstream_dataset_info.json` 来自原始训练数据导出的元信息，记录文件名、源字节数、特征类型和训练/测试规模。

上游当前数据卡标注CC-BY-4.0；请保留SolarBench、SKIPP'D原数据作者及来源署名，并遵循实际使用版本的数据条款。本仓代码GPL许可证不改变上游数据许可。

本仓包含：

- `proc/{split}_meta.npz`：`minutes`、`frame_rows`、`pv`。
- `proc/{split}_samples[_h5|_h30].npz`：`anchors`、`win`、`tgt`、`anchor_time`。
- `proc/{split}_clearsky.npz`：解析晴空功率`p_theo`。
- `proc/day_weather.json`：历史探索性天气标签。
- `manifest.json`：实验数组与图像文件的内容散列。

本地重建但不提交：`raw/*.parquet`、`My_SKIPPD_Data/`、`proc/{split}_images.dat.npy`。

```bash
python scripts/download_data.py
python src/prepare_data.py
python src/prep_clearsky.py
python scripts/verify_assets.py --skip-assets --arrays --images
```

下载约2.32GB；图像物化产生train `(349372,64,64,3)`、test `(14003,64,64,3)` uint8数组，总计约4.47GB。输入窗口使用图像内存映射，避免为每个重叠窗口重复存储图像。

**数据语义**：按照当地完整秒去重，不把带秒偏移的相机帧强行对齐整分钟。15帧需要严格60秒连续，且未来目标时刻确实存在。完整训练来源窗口按天划分，再按stride2采样，不跨日混合训练/验证。

本项目不是官方HDF5 stacks的直接副本；15分钟重建测试窗数为11,180。不要把上游14,003行全部当作本协议forecast样本。
