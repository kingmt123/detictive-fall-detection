# 架构调整建议（基于训练日志 + 论文调研）

> 基于 `run.json`（MultiStreamMultiScaleAttentionTCN, 103K params）和 `history.jsonl`（15 epoch）的诊断，
> 结合端侧跌倒检测/HAR 论文和 GitHub 项目的调研结果。

---

## 一、训练日志诊断

### 实际表现（比预期好得多）

| Epoch | MAP | P@R90 | P@R95 | val_loss | train_loss | LR |
|---|---|---|---|---|---|---|
| 0 | 56.66% | 59.35% | 53.96% | 0.468 | 0.643 | 2.97e-4 |
| **2** | **61.42%** | **62.89%** | **59.94%** | **0.372** | 0.537 | 2.71e-4 |
| 5 | 59.85% | 63.29% | 56.42% | 0.396 | 0.476 | 1.96e-4 |
| 10 | 58.22% | 63.69% | 52.75% | 0.399 | 0.426 | 4.96e-5 |
| 14 | 57.30% | 62.89% | 51.72% | 0.403 | 0.415 | 0 |

**关键发现：MAP 在 epoch 2 达到 61.42%，比之前报告的 ~40% 高了 20pt！** 之后持续下降是训练策略问题，不是架构问题。

### 三个训练问题

| 问题 | 证据 | 影响 |
|---|---|---|
| **LR 衰减太快** | cosine 从 3e-4→0，ep12 已接近 0 | 模型 ep2 后无法继续学习 |
| **15 epoch 太少** | ep2 是最佳，之后全是过拟合 | 没有充分训练 |
| **无早停** | ep2 后连续 12 epoch 退化 | 浪费时间且选了最差 checkpoint |

### 过拟合模式

```
train_loss: 0.643 → 0.415（持续下降 ✓）
val_loss:   0.468 → 0.372 → 0.403（先降后升 ✗）
MAP:        56.66% → 61.42% → 57.30%（先升后降 ✗）
```

**典型过拟合**：训练 loss 还在降，但 val 已经恶化。

---

## 二、训练策略修复（立即可做，预期 MAP +5-10%）

### 修复 1：学习率调度

```python
# 当前：cosine decay 3e-4 → 0（15 epoch 内衰减到 0）
# 修复：warmup + cosine，最小 LR 不低于 1e-5
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=50, eta_min=1e-5  # 50 epoch，最低 1e-5
)
# 或用 OneCycleLR（更适合短训练）
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, epochs=50, steps_per_epoch=len(train_loader)
)
```

### 修复 2：早停

```python
# val MAP 连续 5 epoch 不升则停
early_stopper = EarlyStopper(patience=5, metric='clip_map')
# 保存 best checkpoint，不是 last
```

### 修复 3：更强正则化

```python
# 当前：dropout=0.35, weight_decay=0.0005
# 修复：
config = {
    "dropout": 0.4,           # +0.05
    "weight_decay": 0.001,    # 2×
    "label_smoothing": 0.1,   # 新增
    "grad_clip": 1.0,         # 从 5.0 降到 1.0
    "mixup_alpha": 0.2,       # 新增 mixup 数据增强
}
```

### 修复 4：训练更长 + 更大 LR

```python
config = {
    "epochs": 50,             # 15 → 50
    "learning_rate": 1e-3,    # 3e-4 → 1e-3（更大的初始 LR）
    "batch_size": 256,        # 512 → 256（更频繁更新）
}
```

### 预期效果

| 修复 | 预期 MAP 提升 | 依据 |
|---|---|---|
| LR 调度修复 | +3-5% | ep2 后模型还能继续学习 |
| 早停 | +1-2% | 选到最佳 checkpoint |
| 更强正则化 | +2-3% | 减少过拟合 |
| 更长训练 | +2-3% | 充分收敛 |
| **合计** | **+8-13%** | MAP 有望达到 **65-75%** |

---

## 三、架构微调建议（基于论文调研）

### 当前架构评估

当前 `MultiStreamMultiScaleAttentionTCN` 已经包含：
- ✅ 4 流（joint/bone/motion/geometry）
- ✅ 多尺度卷积核 [3, 5, 7]
- ✅ 扩张卷积 [1, 2, 4, 8]
- ✅ 4 头注意力
- ✅ 因果卷积（不看未来帧）
- ✅ 103K 参数（极轻量）

**架构本身合理，不需要大改。** 但可以微调：

### 微调 1：增大 stream_channels

```python
# 当前：stream_channels=32, stream_output_dim=64
# 调整：stream_channels=64, stream_output_dim=128
# 参数增加：103K → ~400K（仍极轻量）
# 预期 MAP：+2-3%
```

### 微调 2：加 SE 通道注意力

```python
# 在每个 TCN block 后加 SE
# 当前已有 attention_heads=4，但 SE 更轻量且互补
# 参数增加：~5K
# 预期 MAP：+1-2%
```

### 微调 3：加几何特征流

```python
# 当前 geometry_observed=6 维
# 扩展为 10 维：加 torso_angle, hip_height, bbox_ratio, velocity_y
# 这些是 PIFR 论文证明有效的强特征
# 参数增加：~2K
# 预期 MAP：+2-4%
```

### 微调 4：窗口 16→24 帧

```python
# 当前：window_size=16（0.53s @ 30fps）
# 调整：window_size=24（0.8s）
# 跌倒过程 0.5-1.5s，24 帧覆盖更完整
# 参数不变，计算量 +50%
# 预期 MAP：+1-3%
```

---

## 四、参考论文与项目

### 端侧轻量架构

| 论文/项目 | 核心 | 参数量 | 精度 | 链接 |
|---|---|---|---|---|
| Light-MHTCN | 多头 TCN | 轻量 | HAR SOTA | ScienceDirect 2023 |
| LMTCN | 轻量多尺度 TCN | 轻量 | — | sciopen 2024 |
| GTAM-CNN | 分组时序注意力 CNN | 轻量 | HAR | SPIE 2024 |
| Dual-branch TCN | 双分支注意力 TCN | 轻量 | 边缘识别 | ScienceDirect 2025 |
| Multi-channel TCN | 多通道双注意力 TCN | 轻量 | — | Springer 2026 |
| WTCN | 小波 TCN | 轻量 | 99.53% fall | Wiley 2022 |

### 跌倒检测轻量方案

| 论文 | 核心 | 精度 | 链接 |
|---|---|---|---|
| LiteFallNet | 轻量可解释 | 实时 | PMC 2025 |
| Cascade Fall Detection | 粗→细两阶段 | 极端不平衡 | Nature SR 2026 |
| PIFR | 姿态角度规则 | 97% | PMC 2025 |
| WTCN | 小波+TCN | 99.53% | Wiley 2022 |
| Multi-channel Fall | 多通道轻量 | 91-98% | Springer 2026 |

### GitHub 可参考项目

| 项目 | 内容 | 链接 |
|---|---|---|
| qbxlvnf11/skeleton-based-action-recognition-methods | 多方法 PyTorch 实现 | https://github.com/qbxlvnf11/skeleton-based-action-recognition-methods |
| firework8/Awesome-Skeleton-based-Action-Recognition | 论文列表 | https://github.com/firework8/Awesome-Skeleton-based-Action-Recognition |
| kajal1106/Elderly-Fall-Detection-LSTM-CNN-BiLSTM-GRU | 跌倒检测多模型 | https://github.com/kajal1106/Elderly-Fall-Detection-Using-Deep-Learning-Models-LSTM-CNN-Bi-LSTM-and-GRU |
| simplexsigil/omnifall-experiments | OmniFall 官方实验 | https://github.com/simplexsigil/omnifall-experiments |
| open-mmlab/mmaction2 | 动作识别工具箱 | https://github.com/open-mmlab/mmaction2 |

---

## 五、推荐实施路线

### Phase 0（0.5 天）：修复训练策略

```bash
# 不改架构，只改训练参数
# - LR: 1e-3 + OneCycleLR
# - Epochs: 50 + 早停 patience=5
# - Dropout: 0.4, Weight decay: 0.001
# - Label smoothing: 0.1
# - Batch size: 256
# 预期 MAP: 65-75%
```

### Phase 1（1 天）：微调架构

```bash
# - stream_channels: 32→64
# - geometry_features: 6→10（加 torso_angle, hip_height, bbox_ratio, velocity_y）
# - window_size: 16→24
# 预期 MAP: 68-78%
```

### Phase 2（1 天）：数据增强

```bash
# - 随机关节噪声（±0.02）
# - 随机遮挡（随机关节置零）
# - 速度扰动（±20%）
# - Mixup（alpha=0.2）
# 预期 MAP: +2-3%
```

### Phase 3（可选）：级联架构

```bash
# 第一级：规则特征快速筛选（torso_angle + bbox_ratio）
#   - 通过 → 进入第二级
#   - 不通过 → 直接判负
# 第二级：TCN 精细分类
# 优势：减少假阳性，提升 P@R90/P@R95
```

---

## 六、关键结论

1. **架构不是瓶颈**：103K 参数的 4 流 TCN 已经达到 61.42% MAP
2. **训练策略是瓶颈**：LR 调度、早停、正则化可以再提升 8-13%
3. **几何特征是金矿**：PIFR 用 2 个角度就达到 97%，你的 geometry 流只有 6 维，可以扩展
4. **窗口可以更大**：16 帧（0.53s）不够覆盖完整跌倒，24 帧（0.8s）更好
5. **不要加更多流**：4 流已经足够，加更多会过拟合
