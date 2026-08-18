# 论文索引

> 按改进方向分类，包含直接可参考的论文和 GitHub 实现链接。

## 方向 A：扩大 TCN 容量

| 论文 | 链接 | GitHub |
|---|---|---|
| MSTCN: Multiscale TCN for Activity Recognition | https://pmc.ncbi.nlm.nih.gov/articles/PMC9989544/ | https://github.com/sj-li/MS-TCN2 |
| MS-TCN++: Multi-Stage TCN (CVPR 2020) | — | https://github.com/sj-li/MS-TCN2 |
| SE-TCN: Squeeze-Excitation TCN | 见 ANALYSIS.md A3 | — |

## 方向 B：多流特征融合

| 论文 | 链接 | GitHub |
|---|---|---|
| 2s-AGCN: Two-Stream Adaptive GCN (CVPR 2019) | 见论文引用 | https://github.com/littlepure2333/2s_st-gcn |
| 三流 GCN 跌倒检测 (Nature SR 2025) | https://www.nature.com/articles/s41598-025-95508-7 | — |
| CTR-GCN: Channel-wise Topology Refinement (ICCV 2021) | https://github.com/Uason-Chen/CTR-GCN | https://github.com/Uason-Chen/CTR-GCN |
| MS-G3D: Multi-Scale Graph 3D (CVPR 2020) | — | https://github.com/kenziyuliu/MS-G3D |
| HD-GCN: Hierarchically Decomposed GCN (ICCV 2023) | https://github.com/Jho-Yonsei/HD-GCN | https://github.com/Jho-Yonsei/HD-GCN |

## 方向 C：注意力机制

| 论文 | 链接 | GitHub |
|---|---|---|
| TCNTE: TCN + Transformer Encoder (PMC 2025) | https://www.sciencedirect.com/science/article/abs/pii/S1574119225000057 | — |
| ST-TR: Spatial-Temporal Transformer | https://www.researchgate.net/publication/351441247 | — |
| LST-AGCN: Lightweight Attention GCN (2024) | https://www.mdpi.com/2504-2289/10/4/125 | — |
| ReL-SAR: 2.8M params 95.27% accuracy (2024) | https://arxiv.org/html/2409.05749v1 | — |
| InfoGCN: Information Maximization GCN (CVPR 2023) | — | https://github.com/stnoah1/infogcn |

## 方向 D：长时间窗口

| 论文 | 链接 | GitHub |
|---|---|---|
| Skeleton Joint Dynamics Fall Detection (arXiv 2025) | https://arxiv.org/html/2503.06938v1 | — |
| Berkeley ST-GCN Learnable Edges (EECS 2024) | https://www2.eecs.berkeley.edu/Pubs/TechRpts/2024/Archive/EECS-2024-115.pdf | — |
| MoveNet Lightweight Fall Detection (arXiv 2024) | https://arxiv.org/html/2401.01587v1 | — |

## 综合资源

| 资源 | 链接 |
|---|---|
| Awesome-Skeleton-based-Action-Recognition | https://github.com/firework8/Awesome-Skeleton-based-Action-Recognition |
| mmaction2 骨架动作识别工具箱 | https://github.com/open-mmlab/mmaction2 |
| OmniFall 统一跌倒检测基准 | https://arxiv.org/html/2505.19889v3 |
| OmniFall 官方实验 | https://github.com/simplexsigil/omnifall-experiments |
| 多方法 PyTorch 实现 | https://github.com/qbxlvnf11/skeleton-based-action-recognition-methods |
| UP-Fall 3D 骨架数据 | https://github.com/Tresor-Koffi/3D_skeletons-UP-Fall-Dataset |
| NTU RGB+D 数据集 | https://github.com/shahroudy/nturgb-d |

## 新增论文（深度调研补充）

### 跌倒检测直接相关

| 论文 | 链接 | 关键点 |
|---|---|---|
| YOLO11n-pose + 扩张卷积 TCN (SPIE 2025) | https://www.spiedigitallibrary.org/conference-proceedings-of-spie/14119/141190I/ | 直接对标方案 |
| 多阶段跌倒检测 3D 姿态 TCN (Nature SR 2025) | https://www.nature.com/articles/s41598-025-11325-y | 多阶段设计 |
| 跌倒检测 OpenPose + LSTM/GRU (PMC 2021) | https://www.mdpi.com/2076-3417/11/1/329 | LSTM/GRU baseline |

### 轻量边缘部署

| 论文 | 链接 | 关键点 |
|---|---|---|
| Tiny-HAR 边缘 HAR (IEEE 2024) | https://ieeexplore.ieee.org/document/11373235 | IoT 边缘设备 |
| 知识蒸馏骨架轻量模型 (ScienceDirect 2023) | https://www.sciencedirect.com/science/article/abs/pii/S1568494623011845 | KD 压缩 |
| MAG-KD 边缘蒸馏 0.48M (2024) | https://www.researchgate.net/publication/403559863 | 0.48M/94.25% |
| 轻量 GCN 高效骨架识别 (IEEE 2024) | https://ieeexplore.ieee.org/document/10651467 | GCN 轻量化 |

### 图卷积与注意力

| 论文 | 链接 | 关键点 |
|---|---|---|
| DG-STGCN 动态图时空卷积 (arXiv 2022) | https://arxiv.org/abs/2210.05895 | 分组时序+动态融合 |
| SkeletonMAE 自监督预训练 (ICPR 2022) | https://arxiv.org/abs/2209.02399 | 自监督骨架预训练 |
| SGN 语义引导 0.69M (CVPR 2020) | 引自 DenseGCN 综述 | 0.69M/89.0% |
| DSTA-Net 解耦时空注意力 | https://arxiv.org/abs/2007.03263 | 空间+时序解耦注意力 |
| DSTA-Net GitHub | https://github.com/lshiwjx/DSTA-Net | PyTorch 实现 |

### 综述论文

| 论文 | 链接 | 关键点 |
|---|---|---|
| 跌倒检测 SOTA 综述 (ScienceDirect 2025) | https://www.sciencedirect.com/science/article/pii/S2667099225000350 | 全面综述 |
| AI 老人跌倒检测综述 (Springer 2026) | https://link.springer.com/article/10.1007/s12559-026-10550-5 | 最新综述 |
| 跌倒检测系统综述 (PMC 2025) | https://pmc.ncbi.nlm.nih.gov/articles/PMC12609574 | 系统综述 |
| 基于姿态的跌倒检测综述 (PMC 2025) | https://pmc.ncbi.nlm.nih.gov/articles/PMC12107650 | 姿态估计框架对比 |
