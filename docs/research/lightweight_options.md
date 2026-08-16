# 模型轻量化与部署加速技术选型调研报告

> 项目：视觉实时跌倒检测（云端 V100 推理，原端侧 NPU）
> 约束：总参数量 ≤ 20M（越少越加分）；单帧端到端推理 ≤ 100ms（越快越加分）
> 候选：YOLOv8n 检测（3.2M）、YOLOv8n-pose（3.3M）、轻量 TCN/GRU（<1M）
> 调研日期：2026-08-16　所有关键数字均标注来源；未能交叉验证的数字标 **待验证**。

---

## 0. 结论速览（TL;DR）

| 方案 | 检测/姿态 | 时序网络 | 总参数 | V100 预计时延 | 说明 |
|---|---|---|---|---|---|
| **主推（冲精度+达标）** | YOLO11s-pose 9.9M（蒸馏自 m/x-pose）+ imgsz 512 + TensorRT FP16 | 轻量 TCN ≈0.3M | **≈10.2M** | **≤15ms（待验证，余量大）** | 满足 8~12M / ≤40ms 目标 |
| 备选（冲参数加分） | YOLO11n-pose 2.9M（蒸馏）+ imgsz 512 + TensorRT FP16 | TCN ≈0.3M | **≈3.2M** | ≤10ms（待验证） | 参数量极低，精度靠蒸馏+多帧时序补偿 |
| 不推荐 | ST-GCN 全家桶（3.1M+ 且延迟偏高） | — | — | — | 对二分类跌倒任务收益/成本比差 |

---

## 1. 参数量压缩技术对比

### 1.1 结构化剪枝（Structured Channel Pruning）

- **做法**：以 BN 缩放因子 / L1 / Group-Norm 重要度为准则剪除整个通道，迭代"剪枝→微调"恢复精度。工程上 [Torch-Pruning](https://github.com/VainF/Torch-Pruning) 对 YOLOv8/v11 有成熟支持（注意 Detect head 的 Concat/cv2 依赖需特殊处理，社区有踩坑记录：https://github.com/sefaburakokcu/yolo-pruning ）。
- **实测参考**：对 YOLOv11n 做 L1 通道剪枝的头盔检测研究中，剪枝后 **2.378M 参数、 mAP50=0.7882**，优于未剪枝 YOLOv8n（3.011M / 0.7733）——即剪枝+微调可以做到"参数↓20%、精度持平甚至反超"（来源：https://iieta.org/journals/isi/paper/10.18280/isi.310503 ）。
- **经验数据**：剪枝率 ≤30% 时 mAP 损失通常 <1pt；>50% 需蒸馏配合（**待验证**，任务相关）。剪枝对 NMS 后处理无加速，主要减 backbone/neck FLOPs。
- **适用性判断**：本项目检测头只有"人"一个类别，通道冗余大，剪枝收益预期较好；但工程量中等，建议作为第二梯队手段（先蒸馏/重参数化，不够用再剪枝）。

### 1.2 知识蒸馏（Knowledge Distillation）

- **路线**：教师 YOLOv8s/m-pose 或 YOLO11m-pose → 学生 n 级模型。
- **特征图级蒸馏**优于纯 logits 蒸馏，代表作 **MGD（Masked Generative Distillation, ECCV 2022）**：随机掩码学生特征、强迫其"生成"教师完整特征，在检测/分割任务上对小模型提升显著（RetinaNet-ResNet50 学生 +2.1~2.8 AP 量级；来源：https://arxiv.org/abs/2205.01529 ）。YOLOv8 上的 MGD 复现（教师 s → 学生 n）在 PCB 缺陷等定制数据上报告 +1~3 mAP（来源：https://www.researchgate.net/publication/408865759_Lightweight_PCB_defect_detection_via_knowledge_distillation ；具体数值**待验证**，依数据集而定）。
- 现成实现参考：garlic-byte/yolov8_distillation（BN 剪枝 + 蒸馏一体流程，https://deepwiki.com/garlic-byte/yolov8_distillation ）。
- **成本**：仅训练期开销（教师前向），推理零成本——**性价比最高的压缩手段，本项目首选**。
- 蒸馏对象建议：跌倒检测是单类别任务，直接用自训练的 YOLO11s/m-pose 作教师（同数据域，远好于 COCO 预训练教师）。

### 1.3 重参数化（Structural Re-parameterization）

- **原理**：训练期多分支（3×3 + 1×1 + BN），推理期融合成单路 3×3 卷积——**训练精度不减、推理零分支开销**。
- 代表模块：
  - **RepConv / RepVGG 式**：YOLOv6/v7/v10 已内建；Ultralytics 侧可对 backbone 卷积做 RepConv 化（**待验证**官方支持度，社区有 RepNCSPELAN 等移植）。
  - **RepGhost**（CVPRW/arxiv 2211.06088）：用重参数化实现隐式特征复用，替代 GhostNet 的 concat，专为硬件友好设计，号称同精度下更省内存访问（来源：https://arxiv.org/html/2211.06088 ，官方代码 https://github.com/ChengpengChen/RepGhost ）。
- **收益量级**：FLOPs 不变、实际延迟降 10~30%（V100 上分支融合 + 减少内存搬运）；参数量推理态不变（**注意：申报参数量按推理态融合后计算才有效**）。
- **适用性判断**：收益稳定、风险低，建议与蒸馏叠加使用。

### 1.4 量化（FP16 / INT8）

| 精度 | V100 支持 | 典型加速（vs FP32） | 精度影响 | 申报口径风险 |
|---|---|---|---|---|
| FP16 (TensorRT) | 原生 Tensor Core，最佳 | 2~3×（来源：https://github.com/umitkacar/onnx-tensorrt-optimization ） | mAP 损失通常 <0.3pt | 低——权重可按 fp32 存盘申报，推理时转 FP16 |
| INT8 (TensorRT PTQ/QAT) | 支持（V100 有 INT8 Tensor Core，性能约为 FP16 的 2×，**待验证**实测） | 4×+（同上来源）；边缘端实测 YOLOv8n 47ms→15ms（Jetson Orin Nano，来源：https://tildalice.io/onnx-int8-vs-fp16-jetson-orin-nano-latency-benchmark/ ） | **小目标 mAP 掉 5.7pt**（同上来源）；单类别+中大目标场景掉点应更小，**待验证** | **中——需确认评测口径** |

- **关键口径问题**：若评测要求"fp32 权重文件 ≤80MB"，则 20M 参数 × 4B = 80MB 恰好是上限 → 说明**申报按 fp32 存盘计**；量化到 INT8 后文件只有 20MB 但可能不被评测运行时接受（若评测方自行加载 fp32 权重推理，则量化完全不生效）。
- **建议策略**：
  1. 申报/交付物保持 **fp32 .pt/.onnx**（10M 参数 ≈ 40MB，安全达标）；
  2. 自测性能用 **TensorRT FP16**（V100 上收益确定、精度无损）；
  3. INT8 仅当评测规则明确允许量化推理时再启用，且必须配合 QAT 或逐层校准防小目标掉点。

### 1.5 四种技术对比小结

| 技术 | 参数↓ | 延迟↓ | 精度风险 | 工程量 | 本项目优先级 |
|---|---|---|---|---|---|
| 知识蒸馏（MGD/特征级） | —（学生本来就小） | — | **提升**小模型精度 | 低~中 | ★★★★★ |
| 重参数化（RepConv/RepGhost） | 推理态持平 | 10~30% | 几乎无 | 中 | ★★★★ |
| FP16 量化 | 文件减半 | 2~3× | <0.3pt | 极低（一行 export） | ★★★★★ |
| 结构化剪枝 | 20~50% | 10~40% | <1pt（剪枝率≤30%） | 中~高 | ★★★ |
| INT8 量化 | 文件 1/4 | 4×+ | 小目标风险大 | 中（需校准集） | ★★（看评测口径） |

---

## 2. 输入分辨率-精度-时延权衡（640 / 512 / 416）

- **计算量随边长平方缩放**：FLOPs(416) ≈ (416/640)² = 42% of 640；FLOPs(512) ≈ 64% of 640。延迟近似线性于 FLOPs（GPU 上小模型有固定开销，实际降幅略小，**待验证**实测）。
- Ultralytics 官方确认 YOLOv8 全卷积、可变 imgsz 推理，降分辨率直接省算力（来源：https://github.com/ultralytics/ultralytics/issues/5765 ）。
- 第三方对照实验（YOLOv8n 垃圾检测，416 vs 608）：高分辨率 mAP 更高、低分辨率速度更快，符合平方律（来源：https://github.com/Data-pageup/YOLOv8_garbage_detection ）。
- **任务特异性**：跌倒检测中人体占画面比例大（中大目标），416 通常已足够；但**跌倒瞬间人体呈躺卧姿态、长宽比异常**，分辨率过低会损失关键点定位精度（pose 头对分辨率比 detect 头更敏感，**待验证**，建议以关键点评测集为准实测）。
- **推荐策略**：
  - 默认 **512**：预计延迟为 640 的 ~65%，精度损失可控（跌倒场景目标大）；
  - 若 512 下 pose mAP 掉 >1pt，回退 640 纯靠 FP16+蒸馏达标（V100 上 640 也远低于 40ms）；
  - 训练用 640 多尺度、推理用 512 的"训大推小"组合可白赚少量精度（**待验证**，常规经验）。

---

## 3. Ultralytics YOLOv8 / YOLO11 官方与第三方实测时延

### 3.1 官方数据（640 输入，batch=1）

**YOLOv8 检测（COCO）**，来源：https://docs.ultralytics.com/models/yolov8/ （raw 表格核对）

| 模型 | mAP50-95 | CPU ONNX (ms) | A100 TensorRT (ms) | 参数 (M) | FLOPs (B) |
|---|---|---|---|---|---|
| YOLOv8n | 37.3 | 80.4 | 0.99 | 3.2 | 8.7 |
| YOLOv8s | 44.9 | 128.4 | 1.20 | 11.2 | 28.6 |
| YOLOv8m | 50.2 | 234.7 | 1.83 | 25.9 | 78.9 |

**YOLOv8-pose（COCO）**，同上来源：

| 模型 | mAPpose50-95 | A100 TensorRT (ms) | 参数 (M) | FLOPs (B) |
|---|---|---|---|---|
| YOLOv8n-pose | 50.4 | 1.18 | 3.3 | 9.2 |
| YOLOv8s-pose | 60.0 | 1.42 | 11.6 | 30.2 |

**YOLO11 检测（COCO）**，来源：https://docs.ultralytics.com/models/yolo11/ （T4 + TensorRT10 + FP16）

| 模型 | mAP50-95 | CPU ONNX (ms) | T4 TensorRT (ms) | 参数 (M) | FLOPs (B) |
|---|---|---|---|---|---|
| YOLO11n | 39.5 | 56.1±0.8 | 1.5±0.0 | 2.6 | 6.5 |
| YOLO11s | 47.0 | 90.0±1.2 | 2.5±0.0 | 9.4 | 21.5 |
| YOLO11m | 51.5 | 183.2±2.0 | 4.7±0.1 | 20.1 | 68.0 |

**YOLO11-pose（COCO）**，同上来源：

| 模型 | mAPpose50-95 | T4 TensorRT (ms) | 参数 (M) | FLOPs (B) |
|---|---|---|---|---|
| YOLO11n-pose | 50.0 | 1.7±0.0 | 2.9 | 7.4 |
| YOLO11s-pose | 58.9 | 2.6±0.0 | 9.9 | 23.1 |
| YOLO11m-pose | 64.9 | 4.9±0.1 | 20.9 | 71.4 |

### 3.2 第三方实测

- RTX 3080Ti Laptop + TensorRT 10.16 FP16，YOLOv8n 640：检测 **2.46ms**、姿态 **2.28ms**（含 H2D+D2H 全链路，来源：https://github.com/triple-Mu/YOLOv8-TensorRT ）。
- Jetson Orin Nano：YOLOv8n INT8 15ms / FP16 47ms（来源：https://tildalice.io/onnx-int8-vs-fp16-jetson-orin-nano-latency-benchmark/ ）。
- GigaGPU 服务器基准（YOLOv8 n/s/m @640 各 GPU FPS）：https://gigagpu.com/yolov8-nano-small-medium-fps-by-gpu/ （数值**待验证**，可作交叉参考）。

### 3.3 向 V100 / RTX 4060 换算（估算，均标**待验证**，落地前必须用 trtexec 实测）

硬件 FP16 Tensor Core 算力参考：V100 ≈ 125 TFLOPS；T4 ≈ 65 TFLOPS；A100 ≈ 312 TFLOPS；RTX 4060 ≈ 116 TFLOPS（dense，**待验证**精确口径）。

| 模型 | V100 FP16 估算 | RTX 4060 FP16 估算 | 依据 |
|---|---|---|---|
| YOLO11n-pose (640) | **0.8~1.5ms** | **1.5~2.5ms** | T4 实测 1.7ms × 算力比折算；小模型受 kernel launch 固定开销限制，加速比打折 |
| YOLO11s-pose (640) | **1.3~2.5ms** | **2.5~4ms** | T4 实测 2.6ms 折算 |
| YOLO11s-pose (512) | **≈1~2ms** | **≈2~3ms** | 上档 ×0.64（FLOPs 平方律） |

**结论：V100 上即便 640 全精度 FP16，单帧视觉部分也 <5ms，40ms 预算的绝大部分可留给时序网络和前后处理。** 真正的瓶颈会是数据搬运与 Python 前后处理（建议 TensorRT End2End 引擎内嵌 NMS，见 triple-Mu 仓库做法）。

---

## 4. 骨架时序网络选型：TCN vs GRU vs 轻量 ST-GCN（17 关键点序列）

输入规模：17 关节 × (x,y,conf) = 51 维/帧，滑窗 16~32 帧 → 时序网络的计算量与检测器相比**小两个数量级**，选型主要看精度与参数加分，而非时延。

| 网络 | 参数量 | 单窗时延（估算） | 精度特性 | 来源/依据 |
|---|---|---|---|---|
| **TCN**（因果膨胀卷积，4~6 层，通道 64~128） | **0.1~0.5M** | GPU <1ms；CPU <2ms（**待验证**实测） | 并行计算、延迟最低；感受野可覆盖 32+ 帧；对跌倒这种"短时剧烈变化"模式敏感 | TCN+Transformer 用于实时跌倒检测已达实时性要求：https://www.sciencedirect.com/science/article/pii/S1574119225000057 |
| **GRU**（2 层，hidden 128~256） | **0.2~0.6M** | 串行展开 16~32 步，GPU <2ms（**待验证**） | 与 TCN 相当；TCN-GRU 混合架构在跌倒数据集上报告高精度（来源：https://www.sciencedirect.com/science/article/pii/S0167739X22002953 ）；GRU 状态可流式复用，免滑窗重算 | 同上 |
| **轻量 ST-GCN**（原版 ST-GCN ≈3.1M 参数；SGN 0.69M 在 NTU-60 X-Sub 达 89.0%） | 0.7~3.1M | GPU 2~10ms（图卷积 kernel 开销大，**待验证**） | 建模关节间拓扑，对遮挡/关键点噪声鲁棒性最好；但二分类跌倒任务上相对 TCN/GRU 的精度增益通常 <2pt（**待验证**），参数/延迟成本高 | ST-GCN 原文：https://arxiv.org/abs/1801.07455 ；SGN 0.69M/89.0% 数据引自 DenseGCN 对比综述：https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ipr2.12872 ；三流 ST-GCN 跌倒识别（Nature SR 2025）：https://www.nature.com/articles/s41598-025-95508-7 |

**选型结论**：
1. **首选 TCN**：并行卷积、GPU 友好、流式部署时可改成"逐帧因果卷积+环形缓冲"，参数可压到 0.3M 以内；
2. **GRU 作对照基线**（训练快、实现 30 行），若精度打平则取参数更小者；
3. ST-GCN 仅在"多人场景+关键点遮挡严重导致 TCN/GRU 误报"时引入，且选 SGN/EfficientGCN 级轻量变体（<1M）而非原版 3.1M；
4. 时序窗建议 16 帧（@30fps ≈ 0.53s），跌倒动作持续 0.5~1s，兼顾响应速度与上下文。

---

## 5. 落地建议：目标参数 8~12M、时延 ≤40ms 的技术组合

### 5.1 推荐组合（主推方案）

| 组件 | 技术选型 | 参数 | 说明 |
|---|---|---|---|
| 检测+姿态 | **YOLO11s-pose**（9.9M），自数据训练 | 9.9M | 比 v8s-pose 少 1.7M 参数且精度持平/更高（官方表） |
| 蒸馏 | 教师 = YOLO11m-pose 或 x-pose（同数据自训），MGD 特征图级蒸馏 | +0 | 预计 mAP +1~3pt（**待验证**） |
| 分辨率 | 训练 640 多尺度 / 推理 **512** | +0 | 时延 ×0.64 |
| 推理引擎 | **TensorRT FP16 End2End**（NMS 内嵌） | +0 | V100 上视觉部分预计 **2~4ms（待验证）** |
| 时序网络 | **轻量 TCN**（17 关键点 × 16 帧，~0.3M） | ≈0.3M | <1ms |
| 可选加固 | backbone 卷积 RepConv 化 | +0（推理态） | 再省 10~30% 延迟 |

- **总参数量 ≈ 10.2M**（fp32 权重 ≈ 41MB，远低于 80MB 口径上限 ✅）
- **总时延预估**：V100 上 视觉 2~4ms + 时序 <1ms + 前后处理 3~8ms ≈ **≤15ms（待验证，留出 25ms 余量）** ✅
- 满足"参数 8~12M、时延 ≤40ms"双目标，且两项均不踩线、留有余量对冲实测偏差。

### 5.2 备选方案（若评审更看重"参数越少越加分"）

YOLO11n-pose（2.9M）+ MGD 蒸馏 + TCN（0.3M）≈ **3.2M**，V100 时延预计 ≤10ms。牺牲少量姿态精度换取参数量评分，由蒸馏与多帧时序投票补偿。**建议两个方案都训，按评测榜加权决定提交哪个。**

### 5.3 风险与注意事项

1. **评测口径待确认**：fp32 ≤80MB 是否指申报文件格式？量化推理是否被允许？——直接影响 INT8 是否可用，务必先读评测细则（本报告默认保守：fp32 申报 + FP16 推理）。
2. **V100/4060 时延均为折算估算（待验证）**：落地第一周必须用 `trtexec` / `benchmark.py`（triple-Mu 仓库）在真机复测，重点测 512 分辨率下端到端（含预处理）时延。
3. INT8 对小目标掉点明显（文献 -5.7pt），本项目人体为大目标但仍需 QAT 验证，默认不启用。
4. 时序网络输入要用 **归一化关键点 + 置信度加权**，并对 pose 漏检帧做线性插值，否则时序端精度会被检测端噪声主导（**待验证**，工程经验）。

---

## 附：引用来源汇总

- Ultralytics YOLO11 官方参数/时延表：https://docs.ultralytics.com/models/yolo11/
- Ultralytics YOLOv8 官方参数/时延表：https://docs.ultralytics.com/models/yolov8/
- MGD 蒸馏（ECCV 2022）：https://arxiv.org/abs/2205.01529
- RepGhost 重参数化：https://arxiv.org/html/2211.06088 、https://github.com/ChengpengChen/RepGhost
- YOLOv11n 结构化剪枝（头盔检测，2.378M/0.7882）：https://iieta.org/journals/isi/paper/10.18280/isi.310503
- TensorRT FP16/INT8 加速比参考：https://github.com/umitkacar/onnx-tensorrt-optimization
- YOLOv8n INT8 实测（47→15ms，小目标 mAP -5.7pt）：https://tildalice.io/onnx-int8-vs-fp16-jetson-orin-nano-latency-benchmark/
- YOLOv8-TensorRT 实测（3080Ti，2.46ms）：https://github.com/triple-Mu/YOLOv8-TensorRT
- 分辨率-速度权衡官方答复：https://github.com/ultralytics/ultralytics/issues/5765 ；对照实验：https://github.com/Data-pageup/YOLOv8_garbage_detection
- ST-GCN 原文：https://arxiv.org/abs/1801.07455 ；轻量骨架网络对比（SGN 0.69M/89.0%）：https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ipr2.12872
- TCN 跌倒检测：https://www.sciencedirect.com/science/article/pii/S1574119225000057 ；TCN-GRU：https://www.sciencedirect.com/science/article/pii/S0167739X22002953 ；三流 ST-GCN 跌倒：https://www.nature.com/articles/s41598-025-95508-7
- 剪枝工程参考：https://github.com/sefaburakokcu/yolo-pruning ；YOLOv8 蒸馏一体流程：https://deepwiki.com/garlic-byte/yolov8_distillation
