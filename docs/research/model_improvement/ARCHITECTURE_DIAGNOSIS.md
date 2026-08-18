# 架构诊断与改进方案

> 4 流 × 多尺度 TCN × SE × Attention 训练三轮 MAP ~40%，低于基线 42.93%。本文分析原因并给出验证有效的替代方案。

---

## 一、为什么 4 流架构失败？

### 可能原因分析

| 原因 | 证据 | 严重度 |
|---|---|---|
| **过拟合** | 参数从 134K → ~6M（45×），训练数据不变 | 🔴 最可能 |
| **特征噪声** | Bone/Motion/Accel 特征放大了姿态估计噪声 | 🔴 很可能 |
| **训练不充分** | "三轮"可能不够，复杂模型需要更多 epoch | 🟡 可能 |
| **学习率不匹配** | 大模型需要更小的学习率 | 🟡 可能 |
| **缺少正则化** | Dropout、weight decay、label smoothing | 🟡 可能 |
| **特征尺度不一致** | 4 流特征量纲不同，融合困难 | 🟡 可能 |
| **评估协议严格** | P@R90/P@R95 要求召回 90-95% 时保持精度 | 🟠 已知 |

### 关键洞察

**OmniFall 论文（arXiv 2505.19889）的实验显示：**
- 用 OF-Syn 独立训练的 VMAE 模型在 OF-Syn 上只有 45-52% BAcc
- 用 OF-Staged（真实数据）训练才能达到 83-93% BAcc
- **合成数据本身的天花板就不高**

**PIFR 论文（PMC 2025）的启示：**
- 用**简单的姿态角度**就能达到 97% 准确率
- 两阶段方法：先规则筛选，再 ML 分类
- **简单特征 + 简单模型 > 复杂特征 + 复杂模型**

---

## 二、验证有效的架构方案

### 方案 1：规则特征 + 轻量分类器（推荐优先尝试）

**原理**：跌倒的核心特征是**身体角度变化**和**重心下移**，不需要复杂模型。

```python
# 每帧提取 5 个规则特征
features = {
    "torso_angle": angle_between(neck, mid_hip, vertical),  # 躯干倾斜角
    "hip_height": mid_hip.y / frame_height,                  # 髋部归一化高度
    "bbox_ratio": bbox.width / bbox.height,                  # 边界框宽高比
    "velocity_y": (hip.y[t] - hip.y[t-1]) * fps,            # 髋部垂直速度
    "shoulder_hip_dist": dist(shoulder, hip) / body_height,  # 肩髋距离比
}

# 16 帧窗口 → 16×5 = 80 维输入
# 轻量 MLP 或 1D CNN 分类
class RuleFeatureClassifier(nn.Module):
    def __init__(self):
        self.fc = nn.Sequential(
            nn.Linear(80, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 2)
        )
```

**参数量**：~10K（极轻量）
**预期 MAP**：50-60%（基于 PIFR 97% accuracy 推算）
**优势**：不过拟合、可解释、训练快

### 方案 2：单流扩大 TCN（回归简单）

**原理**：当前 134K 太小，但不需要 4 流。单流扩大即可。

```python
# 单流，但更宽更深
class ImprovedTCN(nn.Module):
    def __init__(self, input_dim=51, channels=[128, 256, 256, 512]):
        self.tcn = nn.Sequential(
            # 多尺度输入
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(128, 256, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(256, 256, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(256, 512, kernel_size=3, padding=8, dilation=8),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
        )
        self.se = SEBlock(512)
        self.attn = nn.MultiheadAttention(512, 4, batch_first=True)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(512, 2)
        )
```

**参数量**：~1.5M
**预期 MAP**：48-55%
**优势**：不过拟合、训练快、单流简单

### 方案 3：双流 TCN（Joint + Motion）

**原理**：4 流太多，2 流足够。Joint 提供静态姿态，Motion 提供动态变化。

```python
# Joint 流：关节坐标
joint_stream = TCN(input_dim=51, channels=[128, 256, 256])

# Motion 流：帧间差分（不是单独的 bone/accel，而是简单的帧间差）
motion_stream = TCN(input_dim=51, channels=[64, 128, 128])

# 融合
combined = torch.cat([joint_stream, motion_stream], dim=-1)  # 256+128=384
output = nn.Linear(384, 2)(combined)
```

**参数量**：~2M
**预期 MAP**：48-58%
**优势**：Motion 是最有效的辅助特征，Bone/Accel 噪声大

### 方案 4：规则特征 + TCN 混合（推荐）

**原理**：先用规则特征提供强先验，再用 TCN 学习残差。

```python
# 分支 1：规则特征（5 维 × 16 帧）
rule_features = extract_rule_features(poses)  # [B, 80]
rule_score = MLP(rule_features)                # [B, 2]

# 分支 2：TCN 学习残差
tcn_features = TCN(poses)                      # [B, 256]
tcn_score = nn.Linear(256, 2)(tcn_features)    # [B, 2]

# 融合
final_score = 0.5 * rule_score + 0.5 * tcn_score  # 或学习权重
```

**参数量**：~1M（TCN）+ ~10K（规则）= ~1.01M
**预期 MAP**：52-62%
**优势**：规则特征提供稳定基线，TCN 学习补充

---

## 三、训练策略（比架构更重要）

### 当前可能的训练问题

| 问题 | 解决方案 |
|---|---|
| 学习率太大 | 1e-3 → 1e-4，用 cosine annealing |
| Epoch 不够 | 至少 30-50 epoch |
| 无正则化 | Dropout 0.3 + Weight Decay 1e-4 |
| 无 label smoothing | label_smoothing=0.1 |
| 无数据增强 | 随机裁剪、翻转、噪声、速度扰动 |
| 类别不平衡 | 正负样本 1:2，用 focal loss |
| 无早停 | val MAP 连续 5 epoch 不升则停 |

### 数据增强策略

```python
def augment_skeleton_sequence(poses, p=0.5):
    """骨架序列数据增强"""
    # 1. 随机时间裁剪（±2 帧）
    if random.random() < p:
        offset = random.randint(-2, 2)
        poses = torch.roll(poses, offset, dims=0)
    
    # 2. 随机关节噪声（模拟姿态估计误差）
    if random.random() < p:
        noise = torch.randn_like(poses) * 0.02
        poses = poses + noise
    
    # 3. 随机速度扰动（模拟不同跌倒速度）
    if random.random() < p:
        speed = random.uniform(0.8, 1.2)
        poses = F.interpolate(poses.unsqueeze(0), scale_factor=speed, mode='linear')
    
    # 4. 随机遮挡（模拟遮挡）
    if random.random() < p:
        joint_idx = random.randint(0, 16)
        poses[:, joint_idx, :] = 0
    
    return poses
```

### 损失函数

```python
# Focal Loss 处理类别不平衡
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
```

---

## 四、推荐实施路线

### Step 1（0.5 天）：诊断当前问题

```bash
# 检查训练曲线
# - train loss 是否下降？
# - val loss 是否上升（过拟合）？
# - train MAP vs val MAP 差距多大？
```

### Step 2（1 天）：先试最简单的方案

**方案 1：规则特征 + MLP**
- 实现 `extract_rule_features()`
- 80 维 → MLP → 2 分类
- 预期 MAP：50-60%
- 如果 MAP > 43%，说明规则特征有效

### Step 3（1 天）：单流扩大 TCN

**方案 2：单流 TCN（1.5M 参数）**
- 加 Dropout 0.3 + Weight Decay 1e-4
- 用 Focal Loss
- 数据增强
- 训练 50 epoch + 早停
- 预期 MAP：48-55%

### Step 4（1 天）：混合方案

**方案 4：规则特征 + TCN**
- 规则分支提供基线
- TCN 分支学习残差
- 加权融合
- 预期 MAP：52-62%

### Step 5（1 天）：双流 TCN

**方案 3：Joint + Motion 双流**
- 如果 Step 4 还不够，叠加 Motion 流
- 预期 MAP：55-65%

---

## 五、关键教训

1. **简单 > 复杂**：PIFR 用 5 个角度特征就达到 97% accuracy
2. **数据 > 模型**：OF-Syn 合成数据的天花板本身就不高（OmniFall 论文 45-52% BAcc）
3. **训练 > 架构**：正则化、数据增强、学习率调度比架构选择更重要
4. **诊断 > 盲改**：先看训练曲线，确认是过拟合还是欠拟合
5. **规则先验**：跌倒的物理特征（角度、高度、速度）是强先验，不应忽略

---

## 六、参考论文

| 论文 | 核心启示 | 链接 |
|---|---|---|
| PIFR (PMC 2025) | 姿态角度 + 两阶段 → 97% accuracy | https://pmc.ncbi.nlm.nih.gov/articles/PMC12173392 |
| OmniFall (arXiv 2025) | OF-Syn 天花板 45-52% BAcc | https://arxiv.org/html/2505.19889v3 |
| Body Geometry Fall Detection | 身体几何特征 | https://www.sciencedirect.com/science/article/pii/S1047320321002728 |
| TCN-GRU Hybrid (PMC 2024) | TCN+GRU 混合 | https://pmc.ncbi.nlm.nih.gov/articles/PMC11321576 |
| Data Augmentation for Skeleton Fall | 骨架数据增强 | https://ieeexplore.ieee.org/document/9728415 |
| Class-Imbalanced Fall Detection | 类别不平衡处理 | https://pmc.ncbi.nlm.nih.gov/articles/PMC8512051 |
| Fall Detection Model Scored 94% and Lying | 评估陷阱 | https://towardsdatascience.com/my-fall-detection-model-scored-94-and-it-was-lying-to-me/ |
| Skeleton Normalization | 骨架归一化方法 | https://pmc.ncbi.nlm.nih.gov/articles/PMC9185346 |
| Multi-Stage Fall Detection (Nature SR 2025) | 多阶段框架 | https://www.nature.com/articles/s41598-025-11325-y |
| Pre-Impact Fall Detection | 预冲击检测 | https://pmc.ncbi.nlm.nih.gov/articles/PMC4888654 |
