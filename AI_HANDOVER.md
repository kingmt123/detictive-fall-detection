# AI 执行手册 — Detictive 跌倒检测竞赛项目接手

> 本文档交给 AI 助手，让其按顺序执行所有命令。
> 截止时间：2026-09-01 23:59

---

## 第一步：环境搭建

```bash
# 1. clone 仓库
git clone https://github.com/kingmt123/detictive-fall-detection.git
cd detictive-fall-detection

# 2. 安装依赖（需要 CUDA 版 PyTorch，按 PyTorch 官网对应版本安装）
pip install -r requirements.txt
pip install ruff

# 3. 验证测试
python -m pytest -q
# 预期：121 passed

# 4. 验证代码质量
ruff check .
# 预期：All checks passed!
```

## 第二步：数据准备

项目需要两个数据集：

1. **URFD**：30 个跌倒试次（双视角）+ 40 个 ADL 试次
   - 下载后放入 `data/sources/urfd/`

2. **OF-Syn**：OmniFall 合成数据集，约 9.7GB tar 文件
   - 下载后放入 `data/omnifall/data_files/omnifall-synthetic_av1.tar`

数据下载完成后生成 manifest：

```bash
python tools/build_manifest.py
python tools/prepare_omnifall_events.py
```

生成的 `data/manifest.csv` 约 12,100 行，包含所有视频路径和标签。

## 第三步：验证当前成果

```bash
# 验证规则基线（无需 GPU，只用已提交的结果）
cat reports/urfd_val_r0_metrics.json
# 预期：P@R90=0.80, P@R95=0.80, MAP=80.0%

# 验证 4-clip smoke cache（需要 GPU + 数据）
python -m tools.extract_keypoints \
  --manifest data/manifest.csv --dataset urfd --split val \
  --cache-root runs/pose_cache_smoke --temp-root runs/tmp/pose_cache_smoke \
  --model yolo11n-pose.pt --device 0 --crop auto \
  --clip-id fall-18-cam0 --clip-id adl-02-cam0

python -m tools.extract_keypoints \
  --manifest data/manifest.csv --dataset of-syn --split val \
  --cache-root runs/pose_cache_smoke --temp-root runs/tmp/pose_cache_smoke \
  --model yolo11n-pose.pt --device 0 --crop auto \
  --clip-id fall/fall_ch_026 --clip-id fall/fall_ch_085

# 重跑验证 resume（应该 processed=0, resumed=4）
# 再跑一遍上面两个命令
```

## 第四步：执行下一步开发

按以下顺序执行，每步完成后验证再进入下一步：

### 4.1 TCN window consumer contract（用 canary 的 20 个 NPZ）

在 `models/` 或 `tests/` 中实现：
- 读取 NPZ → `(T, P, 17, 3)` dense array
- 滑动窗口切分 `[T_in, P, 17, 3]` → `[T_out]`
- 归一化：减去 torso center，除以 bbox 尺寸
- padding/mask 正确传播
- clip-level 标签正确分配

验收：`python -m pytest tests/test_tcn_window.py -q` 通过。

### 4.2 全量 pose cache

```bash
# 完整 1,200 OF-Syn val cache（约 59 分钟）
python -m tools.extract_keypoints \
  --manifest data/manifest.csv --dataset of-syn --split val \
  --cache-root runs/pose_cache_val --temp-root runs/tmp/pose_cache_val \
  --model yolo11n-pose.pt --device 0 --crop auto

# 10,800 OF-Syn train cache（约 9.1 小时，后台运行）
python -m tools.extract_keypoints \
  --manifest data/manifest.csv --dataset of-syn --split train \
  --cache-root runs/pose_cache_train --temp-root runs/tmp/pose_cache_train \
  --model yolo11n-pose.pt --device 0 --crop auto

# 验证 resume（应该全部 resumed，零 YOLO）
python -m tools.extract_keypoints \
  --manifest data/manifest.csv --dataset of-syn --split val \
  --cache-root runs/pose_cache_val --temp-root runs/tmp/pose_cache_val \
  --model yolo11n-pose.pt --device 0 --crop auto
```

### 4.3 FallTCN 训练

```bash
# 使用只读 cache 训练（epoch 内零 YOLO）
python -m tools.train_tcn \
  --cache-root runs/pose_cache_train \
  --val-cache-root runs/pose_cache_val \
  --epochs 50 --batch-size 32
```

### 4.4 val 消融 + 阈值选择

完成 rule-only、TCN-only、融合三组消融，选择最优工作点。

### 4.5 1080P/V100 pre-seal 硬门

在最终 test 前完成 1080P 端到端时延和 V100 结果。

### 4.6 test seal

OF-Syn test 只消费一次。URFD test 继续封存。

### 4.7 匿名提交

清除所有身份、绝对路径、用户名、元数据。生成：
- PDF 项目文档（≤200MB）
- MP4 演示视频（≤5min，≤200MB）
- ZIP 提交包（≤200MB）

## 关键约束（必须遵守）

| 约束 | 说明 |
|---|---|
| test split | 不得使用，除非在最终冻结条件下由 seal 约束 |
| C 盘 | 仅剩 ~5.8GB，临时文件必须在 D: |
| OF-Syn archive | 9.7GB，tar handle 必须批次复用 |
| 模型参数 | ≤20M，FP32 ≤80MB，推理 ≤100ms |
| 匿名 | 提交材料不得包含任何身份信息 |
| 数据源 | OF-Syn 不是 subject-independent，不能声称身份独立 |

## 项目结构速查

```
eval/evaluate_manifest.py    批量评测（支持 tar URI）
eval/metrics.py              clip/event 评测指标
eval/benchmark.py            参数量与时延基准
infer.py                     单视频推理入口
pipeline/video_source.py     本地 MP4/tar 统一解析
pipeline/pose_extractor.py   单模型姿态提取
pipeline/pose_cache.py       原子 NPZ cache
pipeline/pose_track.py       多目标跟踪
pipeline/rules.py            姿态物理特征
pipeline/fusion.py           累计下坠融合
pipeline/event_aggregator.py 帧分数到事件
pipeline/inference_engine.py 单模型复用推理
models/tcn.py                0.134M 因果 FallTCN
tools/extract_keypoints.py   cache-first 姿态提取 CLI
tools/build_manifest.py      无泄漏视频 manifest
```

## 详细文档

- `HANDOVER.md` — 完整技术交接
- `DEVLOG.md` — 开发决策日志
- `QUICKSTART.md` — 快速启动
- `README.md` — 项目说明
- `.hermes/plans/` — 执行计划和门禁
- `面向低算力端侧平台基于视觉的实时跌倒检测.docx` — 官方赛题要求
