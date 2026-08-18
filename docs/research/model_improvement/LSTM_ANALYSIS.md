# LSTM/GRU 替代 TCN 方案调研

> 针对将 FallTCN 替换为 LSTM/GRU 的可行性分析，包含论文、GitHub 实现和比赛约束评估。
> 截止 2026-08-18，基于 OF-Syn test MAP=42.93%、参数预算 ≤20M、V100 P95≤100ms 的约束。

---

## 一、LSTM vs TCN 核心对比

| 维度 | TCN | LSTM/GRU | 比赛影响 |
|---|---|---|---|
| 并行训练 | ✅ 完全并行 | ❌ 递归，慢 2-5× | 训练时间影响大 |
| 推理时延 | ✅ 极低，TensorRT 友好 | ⚠️ 递归不利于 GPU 优化 | V100 P95 ≤100ms |
| 长距离依赖 | 需深网络/大 dilation | ✅ 天然记忆门控 | 跌倒 0.5-1.5s |
| 梯度稳定性 | ✅ 无梯度问题 | ⚠️ 梯度消失/爆炸 | 训练稳定性 |
| 参数效率 | 灵活可控 | 门控结构有固定开销 | 参数预算 |
| 空间建模 | 需额外模块 | 需额外模块 | 都需要 GCN |
| 边缘部署 | ✅ ONNX/TensorRT 友好 | ⚠️ 需要展开或自定义 | 端侧推理 |

**结论：TCN 在推理速度和训练效率上优于 LSTM，但 LSTM 在长距离依赖建模上有天然优势。最优方案是 TCN+LSTM 混合或纯 TCN。**

---

## 二、跌倒检测中的 LSTM/GRU 方案

### 1. OpenPose + LSTM/GRU（Applied Sciences 2021）
- **论文**：A Framework for Fall Detection Based on OpenPose and LSTM/GRU
- **链接**：https://www.mdpi.com/2076-3417/11/1/329
- **核心**：OpenPose 提取关键点 → LSTM/GRU 分类
- **精度**：LSTM 98.3%，GRU 96.6%（URFD）
- **参数量**：轻量级设计
- **借鉴**：LSTM 的 baseline 实现，可直接对比

### 2. LSTM + Attention 跌倒检测（CEUR-WS 2023）
- **论文**：Fall Detection with LSTM and Attention Mechanism
- **链接**：https://ceur-ws.org/Vol-3517/paper3.pdf
- **核心**：2D/3D 姿态估计 + LSTM + 注意力机制
- **借鉴**：LSTM 输出加注意力权重，聚焦关键时刻

### 3. CNN-BiLSTM + CBAM 跌倒检测（Applied Sciences 2022）
- **论文**：Research on CNN-BiLSTM Fall Detection Algorithm Based on Attention Mechanism
- **链接**：https://www.mdpi.com/2076-3417/12/19/9671
- **核心**：CBAM-IAM-CNN-BiLSTM，CNN 提取空间特征，BiLSTM 捕捉双向时序
- **借鉴**：CNN 空间特征 + BiLSTM 时序 + CBAM 通道/空间注意力

### 4. 3D CNN + LSTM 跌倒检测（arXiv 2025）
- **论文**：Human Fall Detection using Transfer Learning-based 3D CNN
- **链接**：https://arxiv.org/html/2506.03193v1
- **核心**：预训练 3D CNN 提取特征，LSTM 处理关键点序列
- **借鉴**：迁移学习 + LSTM 的组合策略

### 5. Multi-Modal CNN-LSTM + Multi-Head Attention（arXiv 2025）
- **论文**：A Multi-Modal CNN-LSTM Framework with Multi-Head Attention
- **链接**：https://arxiv.org/html/2603.22313v1
- **核心**：多模态传感器融合 + CNN-LSTM + Multi-Head Attention
- **借鉴**：多模态融合策略，Multi-Head Attention 增强关键时刻

### 6. NTU RGB+D LSTM 跌倒检测（EPSTEM 2024）
- **论文**：A Real-Time Fall Detection Framework Using Vision
- **链接**：https://www.epstem.net/index.php/epstem/article/download/1358/1350/1805
- **精度**：NTU RGB+D 上 93.23% precision, 96.xx% recall
- **借鉴**：NTU RGB+D 数据集上的 LSTM baseline

### 7. 下一代跌倒检测（PMC 2025）
- **论文**：Next-generation fall detection: harnessing human pose estimation
- **链接**：https://pmc.ncbi.nlm.nih.gov/articles/PMC12107650/
- **精度**：95.24% sensitivity, 89.80% specificity, 98.00% accuracy
- **借鉴**：三种姿态估计框架对比，LSTM 分类器

---

## 三、骨架动作识别中的 LSTM 方案

### 8. 时空 LSTM（ST-LSTM）（arXiv 2017）
- **论文**：Skeleton-Based Action Recognition Using Spatio-Temporal LSTM
- **链接**：https://arxiv.org/abs/1706.08276
- **核心**：空间域 + 时序域的 LSTM，分别建模关节间和帧间关系
- **借鉴**：空间 LSTM（关节间）+ 时序 LSTM（帧间）的双流架构

### 9. AGC-LSTM — 注意力图卷积 LSTM（IEEE TPAMI 2020）
- **论文**：An Attention Enhanced Graph Convolutional LSTM Network for Action Recognition
- **链接**：https://ieeexplore.ieee.org/document/8954298
- **核心**：图卷积 + LSTM + 注意力，空间和时序同时建模
- **借鉴**：GCN + LSTM + Attention 的三合一架构

### 10. 双流 LSTM-DSConV（Springer 2024）
- **论文**：Skeleton-based human action recognition using LSTM and depthwise separable CNN
- **链接**：https://dl.acm.org/doi/10.1007/s10489-024-06082-w
- **核心**：LSTM + 深度可分离卷积的双流架构
- **借鉴**：DSConV 降低参数量，LSTM 捕捉时序

### 11. ConvLSTM 骨架活动识别（Edge Hill University）
- **论文**：Skeleton-based human activity recognition using ConvLSTM and guided clustering
- **链接**：https://research.edgehill.ac.uk/en/publications/skeleton-based-human-activity-recognition-using-convlstm-and-guid/
- **核心**：ConvLSTM 将卷积嵌入 LSTM 的门控结构
- **借鉴**：ConvLSTM 同时捕捉空间和时序特征，无需单独的 GCN

### 12. 多流 LSTM 骨架融合（Semantic Scholar）
- **论文**：Skeleton Feature Fusion Based on Multi-Stream LSTM for Action Recognition
- **链接**：https://www.semanticscholar.org/paper/7f9b192dad9f85289016c8a089d2a6a65ed1224a
- **核心**：多流 LSTM 融合骨架特征
- **借鉴**：Joint + Velocity + Acceleration 多流 LSTM

### 13. BiLSTM-CNN 骨架识别（IJITCS 2025）
- **论文**：Lightweight 3DCNN-BiLSTM Model for Human Activity Recognition
- **链接**：https://www.mecs-press.org/ijitcs/ijitcs-v17-n6/v17n6-10.html
- **核心**：3D CNN + BiLSTM，轻量级设计
- **借鉴**：BiLSTM 捕捉双向时序依赖

### 14. Seq2Seq BiLSTM 骨架识别（SAGE 2022）
- **论文**：Seq2seq model for human action recognition based on skeleton and two-layer bidirectional LSTM
- **链接**://journals.sagepub.com/doi/10.3233/AIS-220125
- **核心**：两层 BiLSTM 的 Seq2Seq 模型，轻量级
- **借鉴**：极简 BiLSTM 架构，参数量可控

---

## 四、TCN+LSTM 混合方案

### 15. TCN-GRU 混合模型（PMC 2024）
- **论文**：A hybrid TCN-GRU model for classifying human activities
- **链接**：https://pmc.ncbi.nlm.nih.gov/articles/PMC11321576/
- **核心**：TCN 提取局部时序特征 → GRU 捕捉长距离依赖
- **借鉴**：TCN 做特征提取，GRU 做序列建模，各取所长
- **优势**：TCN 并行训练 + GRU 长距离记忆

### 16. LSTM vs TCN 对比研究（MDPI Sensors 2021）
- **论文**：Comparison between Recurrent Networks and TCN for Skeleton-Based Action Recognition
- **链接**：https://www.mdpi.com/1424-8220/21/6/2051
- **核心**：系统对比 LSTM/GRU/TCN 在骨架动作识别上的性能
- **借鉴**：直接的性能对比数据

---

## 五、GitHub 实现

| 项目 | 链接 | 内容 |
|---|---|---|
| Elderly-Fall-Detection-LSTM-CNN-BiLSTM-GRU | https://github.com/kajal1106/Elderly-Fall-Detection-Using-Deep-Learning-Models-LSTM-CNN-Bi-LSTM-and-GRU | LSTM/CNN/BiLSTM/GRU 跌倒检测，MobiAct 数据集 |
| skeleton-based-action-recognition-methods | https://github.com/qbxlvnf11/skeleton-based-action-recognition-methods | 多方法 PyTorch 实现，含 LSTM |
| Awesome-Skeleton-based-Action-Recognition | https://github.com/firework8/Awesome-Skeleton-based-Action-Recognition | 论文列表，含 LSTM 方向 |
| cnn-lstm topic | https://github.com/topics/cnn-lstm | Mediapipe + LSTM/CNN/ViT 姿态 |
| ST-LSTM | https://arxiv.org/abs/1706.08276 | 时空 LSTM 原始论文 |

---

## 六、推荐架构方案

### 方案 1：TCN+LSTM 混合（推荐）

```
输入：32 帧 × 17 关节
  ├─ Joint 流 ───────────────────────────────────┐
  │   (x, y, conf) × 17 = 51 维                  │
  │   → MultiScale TCN (k=3,5,7) + SE            │ → Concat → SE → FC(1024, 2)
  │   → BiLSTM (hidden=128, layers=2)            │
  │   → Temporal Attention (4 heads)              │
  │   → 256 维输出                                 │
  │                                                │
  ├─ Bone 流 ────────────────────────────────────┤
  │   骨骼向量 = 34 维                              │
  │   → MultiScale TCN + SE                       │
  │   → BiLSTM (hidden=128, layers=2)            │
  │   → Temporal Attention                        │
  │   → 256 维输出                                 │
  │                                                │
  ├─ Motion 流 ──────────────────────────────────┤
  │   帧间速度 = 34 维                              │
  │   → MultiScale TCN + SE                       │
  │   → BiLSTM (hidden=128, layers=2)            │
  │   → Temporal Attention                        │
  │   → 256 维输出                                 │
  │                                                │
  └─ Accel 流 ───────────────────────────────────┘
      加速度 = 34 维
      → MultiScale TCN + SE
      → BiLSTM (hidden=128, layers=2)
      → Temporal Attention
      → 256 维输出
```

**参数估算**：
- TCN 部分（每流 ~1.5M）：~6M
- BiLSTM 部分（每流 ~0.3M）：~1.2M
- 注意力 + 融合：~0.5M
- 总计：~7.7M + YOLO 2.87M = **~10.6M**（53% 预算）

### 方案 2：纯 BiLSTM + Attention（更简单）

```
输入：32 帧 × 17 关节
  ├─ Joint 流：51 维 → BiLSTM(256, 2) → Attention → 256 维
  ├─ Bone 流：34 维 → BiLSTM(128, 2) → Attention → 128 维
  ├─ Motion 流：34 维 → BiLSTM(128, 2) → Attention → 128 维
  └─ Accel 流：34 维 → BiLSTM(128, 2) → Attention → 128 维
  → Concat(640) → SE → FC(640, 2)
```

**参数估算**：~4M + YOLO 2.87M = **~6.9M**（34.5% 预算）

### 方案 3：ConvLSTM（空间+时序一体）

```
输入：32 帧 × 17 关节
  → Reshape to (B, T, 17, 3) 视为 17×3 的"图像"
  → ConvLSTM2d layers (kernel=3, channels=64/128)
  → GlobalAvgPool → FC(128, 2)
```

**参数估算**：~2M + YOLO 2.87M = **~4.9M**（24.5% 预算）

---

## 七、比赛约束评估

| 方案 | 参数量 | V100 时延预估 | MAP 预期 | 训练时间 | 推荐度 |
|---|---|---|---|---|---|
| 当前 TCN | 3.0M | 18.28ms | 42.93% | 短 | 基线 |
| 扩大 TCN + ABCD | ~8.9M | ~22-25ms | 55-65% | 中 | ⭐⭐⭐⭐⭐ |
| TCN+BiLSTM 混合 | ~10.6M | ~25-30ms | 55-65% | 长 | ⭐⭐⭐⭐ |
| 纯 BiLSTM+Attention | ~6.9M | ~20-25ms | 50-60% | 长 | ⭐⭐⭐ |
| ConvLSTM | ~4.9M | ~18-22ms | 48-55% | 中 | ⭐⭐⭐ |

**推荐**：优先用方案 1（扩大 TCN + ABCD），如果精度不够再叠加 BiLSTM。

---

## 八、关键论文清单

### 跌倒检测 LSTM/GRU

| # | 论文 | 年份 | 数据集 | 精度 | 关键技术 |
|---|---|---|---|---|---|
| 1 | OpenPose+LSTM/GRU (MDPI) | 2021 | URFD | 98.3%/96.6% | LSTM/GRU baseline |
| 2 | LSTM+Attention (CEUR) | 2023 | — | — | 注意力增强 LSTM |
| 3 | CNN-BiLSTM+CBAM (MDPI) | 2022 | — | — | CNN+BiLSTM+CBAM |
| 4 | 3D CNN+LSTM (arXiv) | 2025 | — | — | 迁移学习+LSTM |
| 5 | Multi-Modal CNN-LSTM (arXiv) | 2025 | — | — | 多模态+Multi-Head Attn |
| 6 | NTU LSTM (EPSTEM) | 2024 | NTU RGB+D | 93.23% P | NTU baseline |
| 7 | Next-gen fall detection (PMC) | 2025 | 多数据集 | 98% acc | 三种框架对比 |

### 骨架动作识别 LSTM

| # | 论文 | 年份 | 关键技术 |
|---|---|---|---|
| 8 | ST-LSTM (arXiv) | 2017 | 时空 LSTM |
| 9 | AGC-LSTM (IEEE TPAMI) | 2020 | GCN+LSTM+Attention |
| 10 | LSTM+DSConV (Springer) | 2024 | 双流 LSTM+DSConv |
| 11 | ConvLSTM (Edge Hill) | — | ConvLSTM 空间+时序 |
| 12 | Multi-Stream LSTM | — | 多流 LSTM 融合 |
| 13 | 3DCNN-BiLSTM (IJITCS) | 2025 | 轻量 3D CNN+BiLSTM |
| 14 | Seq2Seq BiLSTM (SAGE) | 2022 | 极简 BiLSTM |

### TCN+LSTM 混合

| # | 论文 | 年份 | 关键技术 |
|---|---|---|---|
| 15 | TCN-GRU (PMC) | 2024 | TCN 局部+GRU 长距离 |
| 16 | LSTM vs TCN (MDPI) | 2021 | 系统对比研究 |

---

## 九、实施建议

### 优先级

| 优先级 | 方案 | 预期收益 | 实现难度 | 建议 |
|---|---|---|---|---|
| P0 | 扩大当前 TCN（ABCD） | +12-21% | 低 | ✅ 先做这个 |
| P1 | TCN+BiLSTM 混合 | +2-3%（叠加 P0） | 中 | P0 精度不够时叠加 |
| P2 | 纯 BiLSTM 替代 | +0-2% vs TCN | 中 | 作为消融对比 |
| P3 | ConvLSTM | +1-3% | 高 | 实验性方案 |

### 关键洞察

1. **LSTM 不是更好，而是不同**：TCN 擅长局部特征，LSTM 擅长长距离依赖
2. **混合方案最优**：TCN 提取局部特征 → LSTM 捕捉全局依赖 → Attention 聚焦关键时刻
3. **训练时间是瓶颈**：LSTM 递归特性导致训练慢 2-5×，在 deadline 紧张时优先用 TCN
4. **ConvLSTM 值得关注**：同时建模空间和时序，但实现复杂度高
