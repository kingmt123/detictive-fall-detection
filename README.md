# 2026 人工智能大赛：视觉实时跌倒检测

本仓库为比赛最终交付版本（2026-09-12），以项目所有者指定的最终成果目录为准。
源码、依赖清单、模型权重和展示材料均随仓库提供。

## 最终展示材料

- [抹茶吃不饱_面向低算力平台的实时跌倒检测_参赛作品简介.pdf](show/%E6%8A%B9%E8%8C%B6%E5%90%83%E4%B8%8D%E9%A5%B1_%E9%9D%A2%E5%90%91%E4%BD%8E%E7%AE%97%E5%8A%9B%E5%B9%B3%E5%8F%B0%E7%9A%84%E5%AE%9E%E6%97%B6%E8%B7%8C%E5%80%92%E6%A3%80%E6%B5%8B_%E5%8F%82%E8%B5%9B%E4%BD%9C%E5%93%81%E7%AE%80%E4%BB%8B.pdf)
- [抹茶吃不饱_面向低算力平台的实时跌倒检测_项目文档 (1).pdf](show/%E6%8A%B9%E8%8C%B6%E5%90%83%E4%B8%8D%E9%A5%B1_%E9%9D%A2%E5%90%91%E4%BD%8E%E7%AE%97%E5%8A%9B%E5%B9%B3%E5%8F%B0%E7%9A%84%E5%AE%9E%E6%97%B6%E8%B7%8C%E5%80%92%E6%A3%80%E6%B5%8B_%E9%A1%B9%E7%9B%AE%E6%96%87%E6%A1%A3%20%281%29.pdf)
- [抹茶吃不饱_面向低算力平台的实时跌倒检测_项目视频.mp4](show/%E6%8A%B9%E8%8C%B6%E5%90%83%E4%B8%8D%E9%A5%B1_%E9%9D%A2%E5%90%91%E4%BD%8E%E7%AE%97%E5%8A%9B%E5%B9%B3%E5%8F%B0%E7%9A%84%E5%AE%9E%E6%97%B6%E8%B7%8C%E5%80%92%E6%A3%80%E6%B5%8B_%E9%A1%B9%E7%9B%AE%E8%A7%86%E9%A2%91.mp4)

## 快速运行

在仓库根目录安装 `requirements.txt`，然后运行下方部署命令。将 `input.mp4` 替换为实际视频路径，输出路径应使用尚不存在的文件名。

## 文件完整性

`MANIFEST.json` 覆盖原部署文件及 `show/` 中的最终材料，记录字节数和 SHA-256。
本次整理仅更新仓库首页与清单；推理源码、依赖、权重和展示材料与最终成果目录逐字节一致。

## 部署说明与验证范围

### EdgeFall 无标签七头模型

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
