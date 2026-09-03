# EdgeFall 无标签七头部署候选

该包接收原始、无标注视频，只使用 RGB、视频元数据以及运行时生成的 YOLO 姿态轨迹。
它不会读取类别标签、真实事件区间或真实人物 ID。固定输出为七个 raw logits 的算术平均再取
sigmoid，不使用提交数据调阈值。

## 环境与运行

建议 Python 3.11、CUDA 12.x 和 NVIDIA GPU。安装依赖后，在 ZIP 解压目录执行：

```powershell
pip install -r requirements.txt
python -m tools.infer_seven_head input.mp4 `
  --artifact weights/edgefall_seven_head_label_blind_v1.pt `
  --yolo-checkpoint weights/yolo11n-pose.pt `
  --output result.json `
  --device cuda:0
```

`result.json` 包含 clip 分数、七个成员 logits、完全由姿态运动生成的候选区间和分阶段耗时。
批量提交时应在同一进程复用 `SevenHeadVideoModel`，避免反复加载权重。

## 已验证范围

- OF-Syn val（1,200 clips）：mAP 75.208%，P90 77.907%，P95 72.509%。
- 参数：4.053M，FP32 参数量 15.46 MiB（按共享 YOLO、共享 skeleton 和七个头计）。
- 本机完整 3,960 帧视频：15.45 ms/帧平均耗时。
- 从未为本次部署读取 OF-Syn sealed test 或 URFD test。

V100 的端到端 P95 仍需在比赛指定环境复测；在获得该证据前，本包状态为 deployment
candidate，不能把开发验证 mAP 当作最终比赛成绩。
