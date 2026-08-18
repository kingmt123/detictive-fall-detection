# 模型改进方向综合分析

> 针对当前 FallTCN（134K 参数、16 帧窗口、单流 51 维特征）MAP=42.93% 的瓶颈，调研 ABCD 四个改进方向的论文、GitHub 实现与具体方案。
> 截止 2026-08-18，基于 OF-Syn test MAP=42.93%、参数预算 ≤20M、V100 P95≤100ms 的约束。

---

## 一、现状瓶颈诊断

| 维度 | 当前 | 瓶颈 |
|---|---|---|
| FallTCN 参数 | 134K（占预算 0.67%） | 模型容量严重不足 |
| 时间窗口 | 16 帧 / 0.53s | 跌倒通常 0.5-1.5s，窗口不够 |
| 特征维度 | 单流 51 维（x,y,conf × 17 关节） | 无骨骼/速度/加速度信息 |
| 空间建模 | 无（纯时序卷积） | 关节间空间关系未建模 |
| 注意力机制 | 无 | 长距离时序依赖捕捉能力弱 |
| 误报来源 | lie_down/lying/stand_up | 相似动作区分能力不足 |

---

## 二、方向 A：扩大 FallTCN 容量

### 问题
134K 参数的 3 层 TCN（64→64→128）几乎没有非线性表达能力。20M 预算只用了 15%。

### 参考论文

#### A1. MSTCN — 多尺度时序卷积网络
- **论文**：MSTCN: A multiscale temporal convolutional network for user-independent human activity recognition (PMC 2023)
- **链接**：https://pmc.ncbi.nlm.nih.gov/articles/PMC9989544/
- **核心**：基于 Inception 模型的多尺度扩张可分离卷积，不同大小的滤波器提取多尺度特征，扩大感受野
- **参数量**：轻量级设计，适合边缘部署
- **GitHub**：https://github.com/sj-li/MS-TCN2 （MS-TCN++ 多阶段版本）
- **借鉴**：多尺度卷积核（3,5,7）并行提取不同时间尺度特征

#### A2. MS-TCN++ — 多阶段时序卷积
- **论文**：MS-TCN++: Multi-Stage Temporal Convolutional Network for Action Segmentation (CVPR 2020)
- **GitHub**：https://github.com/sj-li/MS-TCN2
- **核心**：多阶段精炼，每阶段 TCN 对前一阶段输出做修正
- **借鉴**：级联精炼结构，第一阶段粗分类，后续阶段细化边界

#### A3. SE-TCN — 挤压激励时序卷积
- **论文**：Squeeze-and-Excitation based Temporal Convolutional Network (SE-TCN) for RUL prediction
- **核心**：在 TCN 每层后加 SE 注意力块，自动学习通道权重
- **借鉴**：以极低参数代价（~5% 额外参数）提升通道选择能力

### 具体改进方案

```
当前：Conv1d(51, 64, k=3) → Conv1d(64, 64, k=3) → Conv1d(64, 128, k=3) → FC
      参数：134K

改进：多尺度 TCN + SE 注意力 + 加宽通道
      ┌─ Conv1d(51, 128, k=3, d=1) ─┐
      ├─ Conv1d(51, 128, k=5, d=1) ─┤ → Concat → SE → 256
      └─ Conv1d(51, 128, k=7, d=1) ─┘
      → Conv1d(256, 256, k=3, d=2) → SE
      → Conv1d(256, 256, k=3, d=4) → SE
      → Conv1d(256, 512, k=3, d=8) → SE
      → GlobalAvgPool → FC(512, 2)
      预估参数：~2.5M
```

---

## 三、方向 B：多流特征融合

### 问题
单流 51 维只包含关节坐标，缺少骨骼结构和运动信息。跌倒的关键区分特征（身体倾斜角度变化、重心下移速度）无法从坐标直接学到。

### 参考论文

#### B1. 2s-AGCN — 双流自适应图卷积
- **论文**：Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition (CVPR 2019)
- **作者**：Lei Shi, Yifan Zhang, Jian Cheng, Hanqing Lu
- **核心**：Joint 流 + Bone 流，各自用自适应图卷积，最后分数融合
- **NTU RGB+D X-Sub**：88.5%（当时 SOTA）
- **GitHub**：https://github.com/littlepure2333/2s_st-gcn （双流 ST-GCN 实现）
- **借鉴**：双流架构（Joint + Bone），每流独立 TCN，最后加权融合

#### B2. 三流时空 GCN（Nature Scientific Reports 2025）
- **论文**：Fall recognition using a three stream spatio temporal GCN model with adaptive feature aggregation
- **链接**：https://pmc.ncbi.nlm.nih.gov/articles/PMC11950412/
- **核心**：Joint + Motion + Residual 三流，Sep-TCN 降低计算量，自适应特征聚合
- **结果**：UR-Fall 98.97%, ImViA 99.68%, Fall-UP 99.97%
- **借鉴**：三流设计（关节+运动+残差），自适应聚合权重

#### B3. CTR-GCN — 通道级拓扑精炼图卷积
- **论文**：Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition (ICCV 2021)
- **GitHub**：https://github.com/Uason-Chen/CTR-GCN
- **核心**：每个通道学习不同的图拓扑，而非共享固定邻接矩阵
- **NTU RGB+D X-Sub**：92.4%
- **借鉴**：通道级拓扑学习，让模型自动发现对跌倒最敏感的关节连接

#### B4. 三流 GCN 跌倒检测（2025）
- **论文**：Fall recognition using a three stream spatio temporal GCN model with adaptive feature aggregation
- **链接**：https://www.nature.com/articles/s41598-025-95508-7
- **核心**：Joint + Motion + Residual，Sep-TCN，自适应聚合
- **精度**：4 个数据集 98-99%+

### 特征定义

```python
# Joint 特征：关节坐标（已有）
joint = poses[T, 17, 3]  # x, y, confidence

# Bone 特征：相邻关节向量（骨骼长度+方向）
bone = []
for (parent, child) in COCO_SKELETON:
    bone.append(joint[child, :2] - joint[parent, :2])  # 17 bones × 2 = 34 维

# Motion 特征：帧间关节速度
motion = joint[1:] - joint[:-1]  # (T-1) × 17 × 2 = 34 维/帧

# Acceleration 特征：帧间速度变化
accel = motion[1:] - motion[:-1]  # (T-2) × 17 × 2 = 34 维/帧
```

### 多流架构

```
输入：16 帧 × 17 关节
  ├─ Joint 流：(x, y, conf) × 17 = 51 维 → TCN → 128 维
  ├─ Bone 流：骨骼向量 × 16 对 = 34 维 → TCN → 128 维
  ├─ Motion 流：帧间速度 × 17 = 34 维 → TCN → 128 维
  └─ Accel 流：加速度 × 17 = 34 维 → TCN → 128 维
  → Concat(512) → SE → FC(512, 2)
  预估参数：~3M（4 流 × ~0.7M/流）
```

---

## 四、方向 C：轻量注意力机制

### 问题
纯 TCN 的感受野受卷积核大小和层数限制。16 帧窗口内，第 1 帧和第 16 帧的关系需要经过 3 次 k=3 卷积才能覆盖（感受野仅 7 帧），无法直接建模。

### 参考论文

#### C1. TCNTE — TCN + Transformer Encoder（PMC 2025）
- **论文**：A Real-time skeleton-based fall detection algorithm combining TCN with Transformer Encoder
- **链接**：https://www.sciencedirect.com/science/article/abs/pii/S1574119225000057
- **核心**：TCN 提取局部时序特征 → Transformer Encoder 捕捉全局依赖
- **结果**：UP-Fall 99%+，实时运行
- **借鉴**：TCN + Transformer 级联，TCN 做局部特征，Transformer 做全局关系

#### C2. ST-TR — 空间时序 Transformer
- **论文**：Skeleton-based action recognition via spatial and temporal transformer networks
- **链接**：https://www.researchgate.net/publication/351441247
- **核心**：用 Transformer self-attention 建模关节间和帧间关系
- **借鉴**：空间注意力（关节间）+ 时序注意力（帧间）

#### C3. LST-AGCN — 轻量注意力图卷积（2024）
- **论文**：LST-AGCN: A Novel Unified Lightweight Attention-based Graph Convolutional Network
- **链接**：https://www.mdpi.com/2504-2289/10/4/125
- **核心**：轻量注意力 + 图卷积，专为边缘设备设计
- **借鉴**：在 GCN 中嵌入轻量通道/时间注意力

#### C4. ReL-SAR — 表示学习骨架动作识别（2024）
- **论文**：ReL-SAR: Representation Learning for Skeleton Action Recognition
- **链接**：https://arxiv.org/html/2409.05749v1
- **核心**：2.80M 参数达到 95.27% 精度（vs 10.08M 参数的 SOTA）
- **借鉴**：极低参数下的高效表示学习策略

### 轻量注意力方案

```python
# 方案 1：Multi-Head Self-Attention（~65K 参数）
class LightTemporalAttention(nn.Module):
    def __init__(self, d_model=128, n_heads=4):
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):  # x: (B, T, D)
        out, _ = self.attn(x, x, x)
        return self.norm(x + out)  # residual

# 方案 2：Squeeze-Excitation 通道注意力（~8K 参数）
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x):  # x: (B, C, T)
        w = self.fc(x).unsqueeze(-1)
        return x * w

# 方案 3：Temporal Shift + Attention（~30K 参数）
# 先用 Temporal Shift 混合相邻帧信息，再用轻量 attention
```

---

## 五、方向 D：扩大时间窗口

### 问题
16 帧 @ 30fps = 0.53 秒。跌倒动作通常持续 0.5-1.5 秒：
- 开始倾斜：0-0.3s
- 快速下落：0.3-0.8s
- 着地：0.8-1.2s
- 躺稳：1.2-2.0s

16 帧窗口可能只看到"开始倾斜"或"着地"，缺少完整过程。

### 参考论文

#### D1. Modeling Human Skeleton Joint Dynamics for Fall Detection（arXiv 2025）
- **链接**：https://arxiv.org/html/2503.06938v1
- **核心**：使用 250 帧的大时间窗口提取跌倒动态
- **借鉴**：长窗口 + 随机起始点训练，让模型学会在不同阶段检测跌倒

#### D2. Berkeley ST-GCN Learnable Edges（EECS 2024）
- **链接**：https://www2.eecs.berkeley.edu/Pubs/TechRpts/2024/Archive/EECS-2024-115.pdf
- **核心**：25 FPS，可学习边的 ST-GCN
- **借鉴**：配合更长窗口的图卷积

### 窗口扩展方案

```
方案 1：直接扩大窗口
  16 → 32 帧（1.07s @ 30fps）
  参数增加：约 2×（TCN 输入维度不变，但序列更长）
  风险：计算量增加，可能超 100ms

方案 2：Stride 降采样
  32 帧，stride=2 → 实际处理 16 帧，但覆盖 1.07s
  参数不变，计算量不变
  风险：丢失细节帧

方案 3：多尺度窗口
  短窗口 16 帧（局部动作）+ 长窗口 48 帧（全局过程）
  两个 TCN 并行，最后融合
  参数约 2×，但能同时捕捉局部和全局

方案 4（推荐）：32 帧 + stride=1，但用更深 TCN
  32 帧输入，4 层 TCN，dilation=[1,2,4,8]
  感受野 = 1+2+4+8 = 15 帧 × kernel_size = 覆盖 32 帧
  参数约 2-3M
```

---

## 六、综合改进方案与参数预算

### 推荐架构：Multi-Stream Multi-Scale TCN with Attention

```
输入：32 帧 × 17 关节
  │
  ├─ Joint 流 ─────────────────────────────────────┐
  │   (x, y, conf) × 17 = 51 维                    │
  │   → MultiScale TCN (k=3,5,7) + SE              │
  │   → 4 层, dilation=[1,2,4,8]                   │
  │   → Temporal Attention (4 heads)                │
  │   → 256 维输出                                   │
  │                                                  │
  ├─ Bone 流 ─────────────────────────────────────┤
  │   骨骼向量 × 16 对 = 34 维                       │ → Concat → SE → FC(1024, 2)
  │   → MultiScale TCN + SE                         │
  │   → 4 层, dilation=[1,2,4,8]                   │
  │   → Temporal Attention                          │
  │   → 256 维输出                                   │
  │                                                  │
  ├─ Motion 流 ────────────────────────────────────┤
  │   帧间速度 × 17 = 34 维                          │
  │   → MultiScale TCN + SE                         │
  │   → 4 层, dilation=[1,2,4,8]                   │
  │   → Temporal Attention                          │
  │   → 256 维输出                                   │
  │                                                  │
  └─ Accel 流 ─────────────────────────────────────┘
      加速度 × 17 = 34 维
      → MultiScale TCN + SE
      → 4 层, dilation=[1,2,4,8]
      → Temporal Attention
      → 256 维输出
```

### 参数预算

| 组件 | 参数量 | 说明 |
|---|---|---|
| YOLO11n-pose | 2,874,462 | 不变 |
| Joint 流 TCN | ~1,500,000 | 多尺度 + SE + Attention |
| Bone 流 TCN | ~1,200,000 | 输入维度更小 |
| Motion 流 TCN | ~1,200,000 | 同 Bone |
| Accel 流 TCN | ~1,200,000 | 同 Bone |
| 融合层 | ~500,000 | SE + FC |
| **合计** | **~8,500,000** | **占预算 42.5%** |
| FP32 体积 | ~34MB | 远低于 80MB |

### 预期改进

| 改进项 | 预估 MAP 提升 | 依据 |
|---|---|---|
| A: 扩大 TCN + 多尺度 | +5-8% | MSTCN 论文报告的改进 |
| B: 多流特征 | +3-6% | 2s-AGCN/三流 GCN 的贡献 |
| C: 注意力机制 | +2-4% | TCNTE 论文的改进 |
| D: 32 帧窗口 | +2-3% | 更完整的跌倒过程建模 |
| **合计** | **+12-21%** | MAP 有望达到 55-65% |

---

## 七、GitHub 实现汇总

| 项目 | 链接 | 用途 | Stars |
|---|---|---|---|
| **Awesome-Skeleton-based-Action-Recognition** | https://github.com/firework8/Awesome-Skeleton-based-Action-Recognition | 论文列表总览 | 2K+ |
| **st-gcn** | https://github.com/yysijie/st-gcn | ST-GCN 原始实现 | 1.5K+ |
| **2s_st-gcn** | https://github.com/littlepure2333/2s_st-gcn | 双流 ST-GCN | 200+ |
| **CTR-GCN** | https://github.com/Uason-Chen/CTR-GCN | 通道级拓扑精炼 | 500+ |
| **MS-G3D** | https://github.com/kenziyuliu/MS-G3D | CVPR 2020 多尺度图卷积 | 500+ |
| **HD-GCN** | https://github.com/Jho-Yonsei/HD-GCN | ICCV 2023 层次分解 GCN | 200+ |
| **InfoGCN** | https://github.com/stnoah1/infogcn | 信息最大化 GCN | 300+ |
| **InfoGCN++** | https://github.com/stnoah1/infogcn2 | 在线版本 | 100+ |
| **MS-TCN2** | https://github.com/sj-li/MS-TCN2 | 多阶段 TCN | 500+ |
| **mmaction2** | https://github.com/open-mmlab/mmaction2 | 动作识别工具箱 | 3K+ |
| **omnifall-experiments** | https://github.com/simplexsigil/omnifall-experiments | OmniFall 官方实验 | 新 |
| **skeleton-based-action-recognition-methods** | https://github.com/qbxlvnf11/skeleton-based-action-recognition-methods | 多方法 PyTorch 实现 | 300+ |
| **3D_skeletons-UP-Fall-Dataset** | https://github.com/Tresor-Koffi/3D_skeletons-UP-Fall-Dataset | UP-Fall 3D 骨架数据 | 新 |

---

## 八、关键论文清单

### 跌倒检测专用

| # | 论文 | 年份 | 数据集 | 精度 | 关键技术 |
|---|---|---|---|---|---|
| 1 | TCNTE (PMC) | 2025 | UP-Fall | 99%+ | TCN + Transformer Encoder |
| 2 | 三流 GCN (Nature SR) | 2025 | UR-Fall/ImViA/Fall-UP | 98-99% | Joint+Motion+Residual, Sep-TCN |
| 3 | LFD-YOLO (Nature SR) | 2025 | PFDD/FPID | mAP+1.5% | 轻量 YOLO 多尺度融合 |
| 4 | BMR-YOLO (PLOS ONE) | 2025 | 多场景 | — | 复杂环境跌倒检测 |
| 5 | Berkeley ST-GCN (EECS) | 2024 | URFD | — | 可学习边 ST-GCN |
| 6 | MoveNet Fall (arXiv) | 2024 | GMDCSA/URFD | sensitivity 0.92 | 轻量姿态+分类器 |
| 7 | Skeleton Joint Dynamics (arXiv) | 2025 | 多数据集 | — | 关节动态建模，250帧窗口 |
| 8 | OmniFall (arXiv) | 2025 | 15K 视频 | — | 统一基准，16 类标注 |

### 骨架动作识别通用

| # | 论文 | 年份 | NTU X-Sub | 关键技术 |
|---|---|---|---|---|
| 9 | ST-GCN (AAAI) | 2018 | 81.5% | 首个时空图卷积 |
| 10 | 2s-AGCN (CVPR) | 2019 | 88.5% | 双流自适应图卷积 |
| 11 | MS-G3D (CVPR) | 2020 | 91.5% | 多尺度图 3D 卷积 |
| 12 | CTR-GCN (ICCV) | 2021 | 92.4% | 通道级拓扑精炼 |
| 13 | PoseC3D (CVPR) | 2022 | — | 3D 热力图卷积 |
| 14 | HD-GCN (ICCV) | 2023 | 93.0% | 层次分解图卷积 |
| 15 | InfoGCN (CVPR) | 2023 | 93.0% | 信息最大化 |
| 16 | ReL-SAR (arXiv) | 2024 | 95.27% | 2.8M 参数高效表示 |
| 17 | LST-AGCN (2024) | 2024 | — | 轻量注意力 GCN |

---

## 九、实施路线图

### Phase 1（1-2 天）：扩大 TCN + 多尺度
- 扩大通道：64→128→256→512
- 多尺度卷积核：3, 5, 7 并行
- 加 SE 注意力
- 窗口 16→32 帧
- 预期 MAP：+5-8%

### Phase 2（2-3 天）：多流特征
- 实现 Bone/Motion/Accel 特征提取
- 4 流独立 TCN
- 加权融合
- 预期 MAP：+3-6%

### Phase 3（1 天）：注意力增强
- 加 Multi-Head Self-Attention
- TCN + Transformer 级联
- 预期 MAP：+2-4%

### Phase 4（1 天）：消融实验
- 单变量消融：每个改进独立测试
- 参数量 vs MAP 曲线
- 时延验证（V100 P95≤100ms）

### 总参数预算检查
- YOLO：2.87M（不变）
- 新 TCN：~5-6M（4 流 + 注意力）
- 合计：~8-9M（远低于 20M）
- FP32 体积：~34MB（远低于 80MB）
