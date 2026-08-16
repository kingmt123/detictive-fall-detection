# 跌倒检测 SOTA 调研报告（LFD-YOLO / BMR-YOLO / YOLO-fall）

> 数据来源：本地文献素材 `docs/research/_raw/`，未联网。素材中不存在的信息一律标注"素材未覆盖"，未做任何推测性编造。
> 调研日期：2026-08-16

## 0. 素材清单与可信度说明

| 论文 | 素材文件 | 可用性 |
|---|---|---|
| LFD-YOLO（Sci Rep 2025, DOI 10.1038/s41598-025-89214-7） | lfd_yolo.txt（正文完整）、lfd_t3~t7.html（表3~表7完整） | ✅ 正文+表格完整 |
| BMR-YOLO（PLoS One 2025, DOI 10.1371/journal.pone.0335992） | bmr_yolo.txt（正文完整，含表1~7）；bmr_try.html 为 Europe PMC 拦截/JS 占位页，无有效内容 | ✅ 正文完整 |
| YOLO-fall（The Computer Journal 2025, DOI 10.1093/comjnl/bxaf005） | yolofall_ss.json 仅含元数据（标题/作者/期刊/DOI，摘要被出版方隐藏）；yolofall.html 为 Cloudflare 拦截页、yolofall_bohrium.html 为阿里云 WAF 加密挑战页、yolofall_colab.html 为 DDoS-Guard 封禁页 | ⚠️ **仅元数据可用，正文内容素材未覆盖** |

YOLO-fall 已确认的元数据（来自 yolofall_ss.json）：标题 *"YOLO-fall: a YOLO-based fall detection model with high precision, shrunk size, and low latency"*；作者 Xiaoyang Zhang, J. Bai, G. Qiao, Xiao Xiao, L. Meng, Shaogang Hu；期刊 The Computer Journal（2025）。标题本身表明其卖点为"高精度 + 小体积 + 低时延"，其余技术细节素材未覆盖。

---

## 1. 三篇论文核心指标对比表

| 维度 | LFD-YOLO | BMR-YOLO | YOLO-fall |
|---|---|---|---|
| 基线/骨干 | **YOLOv5s**（Darknet53 框架 + C3 + SPPF；论文明确认为 YOLOv5 比 YOLOv8 更轻量） | **YOLOv8n**（C2f + SPPF + 解耦头 + Anchor-Free） | 素材未覆盖（标题提示为 YOLO 系改进） |
| 核心创新模块 | ① **CSRG**（Cross Split RepGhost：SRG bottleneck 替换 C3 bottleneck，Split 分支 + RepGhost + SiLU + 推理阶段结构重参数化）② **EMA** 高效多尺度注意力（并行分支、跨空间信息聚合）③ **WFPN** 加权融合金字塔（FPN+PAN 基础上加 A1/A2 两条跨层连接 + 可学习加权融合）④ **GSConv** 替换 Neck 标准卷积 ⑤ **Inner-WIoU** 损失（WIoU v3 + Inner-IoU 辅助框） | ① **BiFormer** 双层路由注意力（加在 backbone 末端，动态稀疏注意力）② **C2f_RVB**（RepViT Block：深度卷积主路 + SE + 1×1 EConv/Projection 交互路，替换 backbone 的 C2f）③ **MultiSEAM** 注意力检测头（CSMM 深度可分离卷积 + 多尺度融合，专攻遮挡）④ **SIoU** 损失（角度/距离/形状/IoU 四分量，方向感知回归） | 素材未覆盖 |
| 参数量 | **5.67 M**（vs YOLOv5s 7.02 M，−19.2%；vs YOLOv8s 11.1 M，−48.6%） | 素材未覆盖（正文与表格仅给 GFLOPs） | 素材未覆盖（标题称 "shrunk size"） |
| 计算量 | **12.6 GFLOPS**（vs YOLOv5s −21.3%；vs YOLOv8s −56.1%） | **6.5 GFLOPs**（vs YOLOv8n 8.7，−2.2；为对比组最低之一） | 素材未覆盖 |
| mAP@0.5 | FPID **84.6%** / PFDD **92.7%**（较 YOLOv5s +1.7 / +1.5 个百分点；较 YOLOv8s +0.5 / +0.3） | BMR-fall **0.899**（较 YOLOv8n 0.852 +0.047，5 次独立运行 mean±std，配对 t 检验 p<0.001，95% CI [0.038, 0.056]）；URFD **0.935**；Le2i **0.894** | 素材未覆盖 |
| mAP@0.5:0.95 | FPID 48.8% / PFDD 70.2% | BMR-fall 0.557；URFD 0.629；Le2i 0.456 | 素材未覆盖 |
| Precision / Recall | FPID 83.1/77.3%；PFDD 88.4/87.5%（模型 F） | BMR-fall 0.877/0.838 | 素材未覆盖 |
| 时延 / FPS | 素材未覆盖（仅给 GFLOPS，无 FPS/时延实测） | 素材未覆盖（仅给 GFLOPs） | 素材未覆盖（标题称 "low latency"） |
| 数据集 | **FPID**（公开，8416 张日常场景 normal+fall）+ 自建 **PFDD**（7859 张，多角度/多光照/部分遮挡；LabelImg 标注，两类："Person" 与 "Down"；7:1:2 划分；翻转+高斯噪声增强） | 自建 **BMR-fall**（11,000 张真实跌倒视频抽帧，LabelMe 标注；**约 70% 人群遮挡场景、30% 低光场景**）+ **URFD**（70 个序列：30 跌倒 + 40 日常活动，地面+天花板双机位）做交叉验证 + **Le2i** 做泛化测试 | 素材未覆盖 |
| 实验环境 | RTX 3080-10G / CUDA 11.8 / PyTorch 2.0 / 640×640 / batch 16 / 300 epochs / SGD | RTX 2080 Ti / Ubuntu 18.04 / CUDA 11.4 / PyTorch 2.1.1 / 640px / batch 16 / 300 epochs | 素材未覆盖 |
| 发表 | Scientific Reports 15:5069, 2025-02-11 | PLoS One 20(11):e0335992, 2025-11-07 | The Computer Journal, 2025（bxaf005） |

### 1.1 LFD-YOLO 消融要点（表4/表5，FPID / PFDD）
- **CSRG**：参数 −13.8%、GFLOPS −17.5%，mAP0.5 +0.5 / +0.4 → 降算力且涨点。
- **EMA**：零额外计算量，mAP0.5 +0.2 → 聚焦人体姿态、抗环境干扰。
- **WFPN**：算力略增，mAP0.5 +0.5 / +0.6 → 跨层连接+加权融合提升多尺度融合效率。
- **GSConv**：精度略降，但参数 −7.4%、GFLOPS −6.0 → 纯轻量化手段。
- **Inner-WIoU**：mAP0.5 +0.5 / +0.3 → 优化中心点定位、适应人体尺度变化。

### 1.2 BMR-YOLO 消融与对比要点
- 注意力机制横评（同条件）：BiFormer mAP@0.5 0.879，优于 SimAM 0.846 / Triplet 0.857 / MPCA 0.863 / MLCA 0.855，且 GFLOPs 8.3 仅微增。
- 全组合（BiFormer + C2f_RVB + MultiSEAM + SIoU）：mAP@0.5 0.899，较基线 +5.4%（相对值），GFLOPs 8.7→6.5。
- 主流模型横评（BMR-fall）：BMR-YOLO 0.899 > YOLOv9t 0.864 > YOLOv10n 0.857 > YOLOv5s 0.855 > YOLOv8n 0.852 > YOLOv7-tiny 0.819 > YOLOv3-tiny 0.724。

---

## 2. "跌倒 vs 相似日常动作"（坐/弯腰/躺/蹲）的处理手段

**LFD-YOLO**：
- 数据层面只做**二分类**（"Person" 正常活动 / "Down" 跌倒），FPID 与 PFDD 均含日常活动样本作为负例。
- 论文在 Discussion 中**明确承认局限**："许多人体动作与跌倒行为相似，可能导致误报（false positives）"，并提出改进方向是"扩展模型以检测更多人体动作类别、更准确地区分相似动作"。即该文本身**没有解决**跌倒 vs 坐/弯腰/躺/蹲的细粒度区分，仅将其列为未来工作。

**BMR-YOLO**：
- 数据层面：URFD 含 40 个日常活动序列作为对照（30 跌倒 + 40 日常），用地面+天花板**双视角**缓解单视角下姿态歧义；检测标签仍以跌倒框为目标。
- 模型层面没有针对"相似动作"的专门分类设计，其抗干扰手段（BiFormer 全局注意力、MultiSEAM 遮挡头、SIoU 回归）主要面向**遮挡与低光**，间接降低误报。对"坐/弯腰/躺 vs 跌倒"的显式区分机制：素材未覆盖。

**YOLO-fall**：素材未覆盖（仅元数据）。

**小结**：两篇可用素材均为"外观目标检测 + 二分类"路线，对相似动作的区分主要依赖数据集里包含日常动作负例，均承认/暗示单帧外观检测难以根治相似动作误报——这正是我们三通道方案中引入 pose+TCN 时序通道与物理规则通道的直接动机。

---

## 3. 低光 / 红外 / 遮挡场景论述

**LFD-YOLO**：
- 数据集：PFDD 刻意覆盖**不同光照条件**与**人体部分遮挡**（多机位角度）。
- 模型：EMA 注意力通过跨空间信息聚合"降低环境干扰影响"，提升遮挡下的人体姿态聚焦；Grad-CAM 场景测试（图13/14）显示：低光下 YOLOv5s 会把周围物体误检为人，LFD-YOLO 在各光照下稳定检出跌倒；部分遮挡下 LFD-YOLO 仍保持高检测精度。
- 红外（IR）：素材未覆盖。

**BMR-YOLO**（三篇中对复杂环境着墨最多）：
- 定位即"complex environments"：摘要与引言明确针对**遮挡 + 低光**两大痛点；批评现有算法"主要关注简单场景，忽视低光照与人群遮挡"。
- 数据集：BMR-fall 约 **70% 人群遮挡、30% 低光**的显式配比，填补公开数据集在挑战性场景上的空白。
- 模型：BiFormer 双层路由注意力抑制背景噪声、动态分配算力到关键区域；MultiSEAM 检测头专为遮挡目标设计（源自 YOLO-FaceV2 的遮挡感知人脸检测）；SIoU 提升遮挡/尺度变化下的回归稳定性。定性实验（图7）显示低光与人群遮挡下 BMR-YOLO 显著减少漏检与误检。
- 红外（IR）：素材未覆盖。

**YOLO-fall**：素材未覆盖（仅元数据）。

**小结**：两篇论文均**未涉及红外/热成像模态**；低光问题的解法均为"可见光 + 数据配比 + 注意力机制"，没有多模态融合设计。若竞赛场景含夜间/红外需求，属三篇共同空白。

---

## 4. 对三通道方案（YOLOv8n 外观 + pose+TCN 骨架时序 + 物理规则）的改进建议

> 前提：外观通道基线为 YOLOv8n，与 BMR-YOLO 同基线，因此 BMR 的改进可直接迁移；LFD-YOLO 的模块虽基于 YOLOv5s 验证，但模块本身（CSRG/EMA/WFPN/GSConv/Inner-WIoU）与版本弱相关，可移植。

1. **外观通道 backbone 末端加 BiFormer 双层路由注意力**（借鉴 BMR-YOLO）。同基线（YOLOv8n）实测 mAP@0.5 0.852→0.879，且 GFLOPs 8.7→8.3 不升反降；动态稀疏注意力把算力分配到人体关键区域、抑制背景噪声，直接服务低光/杂乱背景场景。横评中优于 SimAM/Triplet/MPCA/MLCA，是性价比最高的单点改进。

2. **检测头引入 MultiSEAM 遮挡感知注意力**（借鉴 BMR-YOLO）。跌倒场景中人体常被床/桌椅/他人部分遮挡，MultiSEAM 本就是面向遮挡目标（源自遮挡感知人脸检测 YOLO-FaceV2）设计；BMR 消融显示其对 mAP 提升贡献明确，可显著降低遮挡漏检——同时给 pose 通道提供更完整的检测框，间接提升骨架提取质量。

3. **回归损失由 CIoU 换成 SIoU 或 Inner-WIoU**（分别借鉴 BMR-YOLO / LFD-YOLO）。跌倒姿态导致 bounding box 长宽比与尺度剧烈变化：SIoU 的角度+方向感知分量加速收敛、提升定位稳定性；Inner-WIoU 用缩放因子生成辅助框适配人体尺度变化并抑制极端样本梯度（LFD 消融 mAP +0.3~0.5）。两者可二选一先消融，成本低、收益确定。

4. **轻量化改造：CSRG 思路替换 C2f / Neck 用 GSConv**（借鉴 LFD-YOLO）。CSRG（Split 分支 + RepGhost + 推理阶段结构重参数化）在几乎不掉点的情况下参数 −13.8%、GFLOPS −17.5%；Neck 换 GSConv 再省 参数 −7.4% / GFLOPS −6.0%。省出的算力预算正好补贴给 pose+TCN 通道，保障端侧实时性。WFPN 的跨层加权融合（mAP +0.5~0.6，算力略增）可作为精度优先时的备选。

5. **EMA 注意力增强人体姿态聚焦**（借鉴 LFD-YOLO）。零额外计算量 mAP +0.2，通过并行分支跨空间聚合降低环境干扰，在低光/遮挡下提升外观通道稳定性；且其"聚焦人体姿态细节"的特性与我们 pose 通道形成互补——外观特征越聚焦人体，后续骨架时序判别的输入质量越高。

6. **外观通道从二分类扩展为多动作类别**（借鉴 LFD-YOLO 局限性讨论的建议方向）。LFD 明确承认 Person/Down 二分类无法区分相似动作并建议"扩展到更多动作类别"。建议外观通道直接标注 坐/弯腰/躺/蹲/跌倒 等多类，把"易混淆动作区分"前移到检测层，为 TCN 时序通道和物理规则通道提供动作先验，降低整条链路的误报压力。

7. **数据集构建与配比按复杂场景显式设计**（借鉴 BMR-YOLO + LFD-YOLO）。自建数据按 BMR-fall 的"约 70% 遮挡 / 30% 低光"思路显式配比挑战性样本；按 LFD 的 PFDD 思路覆盖多角度、多光照、部分遮挡，7:1:2 划分 + 翻转/高斯噪声增强。评测上效仿 BMR 用 URFD/Le2i 等公开集做交叉验证与泛化测试，并用 5 次独立运行 mean±std + 配对 t 检验报告显著性。

8. **时序与物理规则通道的定位由素材反向验证**。两篇 SOTA 均承认单帧外观检测对相似动作误报无解（LFD 明确写入 limitation），且都没有时序建模——这恰好是我们 pose+TCN（时序运动模式）+ 物理规则（质心下坠速度、身体倾角突变）通道的差异化价值所在；竞赛答辩中可引用此点论证三通道设计的必要性。

9. **共同空白提示（风险与机会）**：三篇均素材未覆盖红外模态、均无 FPS/端到端时延实测（仅 GFLOPs/GFLOPS）。若竞赛评审关注端侧实时指标，我们应补齐 FPS/时延实测（这是我们的报告能比论文多给出的硬指标）；若有夜间场景，红外/低照度增强是需要自行探索的方向，SOTA 无可借鉴方案。

---

## 5. 参考素材索引

- LFD-YOLO：`_raw/lfd_yolo.txt`（正文）、`_raw/lfd_t3.html`（实验环境）/ `lfd_t4.html`（FPID 消融）/ `lfd_t5.html`（PFDD 消融）/ `lfd_t6.html`（FPID 对比）/ `lfd_t7.html`（PFDD 对比）
- BMR-YOLO：`_raw/bmr_yolo.txt`（正文含全部表格）；`bmr_try.html` 无有效内容
- YOLO-fall：`_raw/yolofall_ss.json`（仅元数据）；`yolofall.html`（Cloudflare 拦截）、`yolofall_bohrium.html`（阿里云 WAF 加密页）、`yolofall_colab.html`（DDoS-Guard 封禁）均无正文内容。**建议后续联网补取该文正文后更新本报告第 1/2/3 节。**
