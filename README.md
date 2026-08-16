# 视觉实时跌倒检测竞赛项目

面向低算力视觉平台的纯视觉跌倒检测原型。目前已完成无需训练即可运行的
**YOLO11n-pose + 多目标跟踪 + 时序物理规则 + 事件聚合**端到端基线，并保留
FallTCN 接口用于后续监督训练。

## 当前能力

- 输入普通 RGB、灰度复制三通道或 URFD `depth|RGB` 横向拼接视频；灰度兼容不等于已完成红外场景验收；
- 检测人体姿态并为每个 track 独立维护时序状态；
- 同时利用累计质心下坠与躯干倾斜，抑制“走向摄像机”的假下坠；
- 输出包含 `track_id` 的跌倒事件 JSON 和带骨架、分数、告警横幅的 MP4；
- 支持 clip-level 与代理 event-level P@R90/P@R95/MAP；
- 构建 URFD 按试次分组、OF-Syn 按官方随机 split 的 manifest；不混用 test 调参。

> 当前规则模型是可展示基线，不是最终竞赛模型。官方尚未明确事件匹配协议；
> `eval/metrics.py` 的 event 模式使用 temporal IoU=0.3 的显式代理假设。

## 环境

- Windows 10 / Python 3.11
- RTX 4060 Laptop 8GB
- PyTorch 2.13.0+cu126
- Ultralytics 8.4.120

CUDA 版 PyTorch 建议按 PyTorch 官方对应索引安装；其余依赖：

```bash
uv pip install -r requirements.txt
```

## 数据准备

项目使用：

- OmniFall 标注及 OF-Syn 12,000 个合成视频；
- URFD 30 个跌倒试次（双视角）和 40 个 ADL 试次。

数据目录被 `.gitignore` 排除。下载完成后生成 manifest：

```bash
python tools/build_manifest.py
python tools/prepare_omnifall_events.py
```

`data/manifest.csv` 保留 `group_id/subject/trial/camera/split`，同一 URFD 试次的多机位
只进入同一个集合。OF-Syn 使用上游随机 clip split；因缺少可靠 subject/scene ID，
不能声称身份独立。

## 运行端到端 Demo

```bash
python infer.py data/sources/urfd/fall-01-cam1.mp4 \
  --output-video reports/demo_fall01_final.mp4 \
  --output-events reports/demo_fall01_final.json
```

已实跑验证结果：

- 输入 160 帧，30 FPS；
- 检出 track 3 事件约 `3.70s–3.80s`；
- clip score 约 `0.64`；
- 输出 MP4 为 320×240、160 帧，可正常解码。

ADL 透视误报回归样本：

```bash
python infer.py data/sources/urfd/adl-01-cam0.mp4 \
  --output-video reports/demo_adl01.mp4 \
  --output-events reports/demo_adl01.json
```

修复“走向摄像机导致质心下移”的问题后，该样本不再输出跌倒事件。

## 测试

```bash
python -m pytest -q
```

当前 **74 个测试**覆盖：评测指标、事件聚合、标签语义、manifest 划分、姿态规则、
多目标跟踪、融合、TCN 形状/参数/因果性。

## 性能基准

```bash
python eval/benchmark.py --warmup 20 --runs 100 \
  --output reports/benchmark_rtx4060.json
```

本机 RTX 4060、640 输入、PyTorch eager 实测（30 次）：

- YOLO11n-pose：2.874M 参数；
- fp32 参数体积估算：11.50MB；
- P50：47.92ms；P95：52.82ms。

这是本机 `model.predict()` 微基准，不含完整解码、跟踪、规则/TCN、聚合、绘制与编码，
也不代表赛事 V100/TensorRT 结果。最终提交前需在 V100 上按官方环境重新测量端到端
P50/P95。

## 项目结构

```text
eval/metrics.py              clip/event 两种评测口径
eval/benchmark.py            参数量与 warm-up 后时延
infer.py                     视频到事件 JSON/可视化 MP4
models/tcn.py                0.134M 因果 FallTCN
pipeline/pose_track.py       多目标姿态跟踪与生命周期
pipeline/rules.py            姿态物理特征
pipeline/fusion.py           累计下坠与姿态融合
pipeline/event_aggregator.py 帧分数到事件
tools/build_manifest.py      无泄漏视频 manifest
tests/                       自动化测试
```

## 下一阶段

1. 先重构可复用、`render=False` 的推理引擎，在完整 URFD val/test 报告规则基线；
2. 实现普通 MP4 + OF-Syn tar 的统一读取和可断点 pose cache；
3. 在小规模缓存 smoke 通过后才训练 FallTCN；
4. 加入低光、灰度/红外模拟和遮挡增强；
5. 经消融后再决定是否加入独立外观 YOLO 通道；
6. 确认官方评测与 NPU 20MB 口径后冻结提交格式。

## 当前限制

- 尚无完整验证集 P@R90/P@R95/MAP；现有 fall-01/adl-01 只是 smoke test。
- 已验证 1920×1080 输入预处理兼容性，但尚未完成 1080P 视频端到端时延基准。
- `eval/benchmark.py` 的 47.92ms 是 YOLO predict 微基准，不是完整端到端告警时延。
- 多目标跟踪采用轻量 IoU + 常速度中心预测，没有 ReID；首次交叉、复杂遮挡和长时间离场仍可能造成 ID 碎片。
- FallTCN 只有网络结构和单元测试，尚无关键点缓存、训练 checkpoint 或真实指标。
