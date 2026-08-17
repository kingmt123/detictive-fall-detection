# 视觉实时跌倒检测竞赛项目 — 交接文档

> **更新时间**：2026-08-17
> **master HEAD**：`e9496d73af253d849ad0c1e087cbf505557a41ea`
> **状态**：Gate 2 全部完成并合并，20-clip canary 通过
> **下一门**：Gate 3 — NPZ→TCN window consumer 验证 → 全量 cache → TCN 训练

---

## 1. 当前完成状态总览

| 项目 | 当前状态 | 证据 |
|---|---|---|
| URFD val 规则基线 | 14/14 完成 | P@R90=0.80, P@R95=0.80, 本地 clip MAP=80.0% |
| Test split | 未使用 | 持久化 one-shot seal；pose cache 和 evaluator 阶段直接拒绝 |
| 普通 MP4 | 支持 | `pipeline/video_source.py` |
| OF-Syn tar URI | Pose cache + rule evaluator 均支持 | `tar://archive!/member`，批次内复用 tar handle |
| Pose cache | 支持 | 原子 NPZ、严格 dtype/shape、源内容 SHA、提取签名 |
| Cache-first 提取 | 支持 | `tools/extract_keypoints.py`，单模型复用 |
| 4-clip smoke | 4/4 成功 | 2 URFD + 2 OF-Syn；重跑 4/4 resume，零 YOLO |
| **20-clip canary** | **20/20 成功** | 4 URFD + 16 OF-Syn；重跑 20/20 resume，零 YOLO |
| 自动化测试 | **121 passed** | `python -m pytest -q` |
| FallTCN | 仅结构/单测 | 无正式 train cache、checkpoint、指标 |

---

## 2. 20-clip canary 实测结果

### 2.1 数据

| 指标 | 值 |
|---|---|
| URFD 首跑 4 clips | 13.76s, 148KB cache |
| OF-Syn 首跑 16 clips | 47.69s, 645KB cache |
| 合计 clips/s | 0.33 |
| Cache/clip 均值 | ~39.7KB |
| 最大单 clip（19人 stand_up）| 271KB |
| Positive:Negative | 10:10 |
| URFD:OF-Syn | 4:16 |

### 2.2 外推

| 场景 | 预估时间 | 预估存储 |
|---|---|---|
| 1,200 OF-Syn val | ~59 min | ~47.6MB |
| 10,800 train | ~9.1h | ~429MB |

### 2.3 预算

- D: 剩余 ~83GB → 磁盘余量 ≥30%
- 截止 2026-09-01 → 时间余量 ≥30%
- **结论：全量 10,800 train + 完整 1,200 val cache，无需冻结子集**

### 2.4 冻结的 canary 集合

```
manifest SHA-256: fdea91af1e226a8fda9c863c1484af8a65841ba68d76cda57f2be9bdc45732df
```

URFD: `fall-01-cam0`, `fall-01-cam1`, `fall-18-cam0`, `fall-25-cam1`,
`adl-02-cam0`, `adl-05-cam1`, `adl-12-cam0`, `adl-40-cam0`

OF-Syn (16): `fall/fall_ch_026`, `fall/fall_ch_043`, `fall/fall_ch_063`,
`fall/fall_ch_085`, `fall/fall_ch_117`, `fall/fall_ch_167`,
`fall/fall_ch_180`, `fall/fall_ch_221`, `stand_up/stand_up_ch_012`,
`stand_up/stand_up_ch_041`, `stand_up/stand_up_ch_047`,
`stand_up/stand_up_ch_114`, `stand_up/stand_up_ch_124`,
`stand_up/stand_up_ch_138`, `stand_up/stand_up_ch_175`,
`stand_up/stand_up_ch_186`

---

## 3. Gate 2 实现（已完成）

### 3.1 统一视频源

`pipeline/video_source.py`：

- cache miss 时本地 MP4 复制到显式 D: temp root 的不可变快照，SHA-256 与 decoder 消费同一份字节；
- tar 成员流式写入调用方指定的 D: 临时目录；
- 一个 `VideoSourceResolver` 在整个批次复用 tar handle；
- 正常退出和异常退出都删除临时视频；
- resume 先检查廉价源身份，再流式检查完整内容 SHA-256；tar resume 不落临时文件。

### 3.2 Pose cache schema v1

每个 clip 一个 NPZ，`allow_pickle=False`：

- `frame_indices`: `int64 [T]`，从 0 连续；
- `timestamps`: `float64 [T]`，等于 `frame_indices/fps`；
- `keypoints`: `float32 [T,P,17,3]`；
- `bboxes`: `float32 [T,P,4]`；
- `track_ids`: `int64 [T,P]`，有效 track 位于每帧前缀、唯一严格递增，padding ID=-1；
- `valid_mask`: `bool [T,P]`；
- `frame_size=[height,width]`；
- 元数据：源身份、源内容 SHA-256、模型/源码/依赖/参数签名、`max_frames`。

写入：同目录临时文件 → flush/fsync → 自校验 → `os.replace`。旧 cache 只有在新 cache 完整写成后才被替换。

### 3.3 提取签名

`PoseExtractor.cache_signature()` 绑定：

- 模型权重 SHA-256（签名和懒加载构造器消费同一个不可变权重快照）；
- `pose_extractor.py`、`pose_track.py` 和实际裁剪实现源码；
- NumPy/OpenCV/PyTorch/Ultralytics 版本；
- device、image size、confidence、crop、`max_frames`；
- `cache_schema` 版本。

100% cache hit 时不构造 YOLO；全 cache miss 后整个批次只加载一次。

### 3.4 Rule evaluator tar 接入

`eval/evaluate_manifest.py`：

- local manifest 路径保持原 `Path` 直通 engine；
- tar URI 经批次级共享 resolver 流式 materialize 后交给 `InferenceEngine`；
- 非法 tar URI 被逐 clip 隔离，记录 `status=error` + `source_kind=unknown`；
- JSONL 记录 `source_kind` 和 `source_prepare_seconds`；
- evaluator run signature 纳入 `video_source.py`；
- CLI `--temp-root` 参数指定临时目录。

---

## 4. 提交历史

```text
master@e9496d7 (HEAD)
├── e9496d7 fix: isolate bad tar URI as per-clip error       ← Gate 2B P1 fix
├── 45127cd feat: evaluate tar-backed manifest videos         ← Gate 2B
├── a52f37d docs: close pose cache implementation gate        ← Gate 2A merge
├── f5f1ea0 fix: bind pose caches to consumed bytes           ← Gate 2A P1 fix
├── bc3ad28 fix: enforce deterministic pose track rows        ← Gate 2A
├── a860a2d feat: harden pose cache resume telemetry
├── 8e4539f feat: extract resumable pose caches from manifest
├── 76f5ef0 feat: add atomic validated pose cache schema
├── 788a077 feat: add unified local and tar video source
└── 1624ec0 docs: define pose cache gate and acceptance criteria
```

---

## 5. 环境要求

| 依赖 | 版本 |
|---|---|
| Python | 3.11 |
| PyTorch | 2.13.0+cu126 |
| torchvision | 0.28.0+cu126 |
| ultralytics | 8.4.120 |
| opencv-python | 5.0.0.93 |
| numpy | 2.4.3 |
| pytest | 9.1.1 |
| PyYAML | 6.0.3 |

安装：

```bash
uv pip install -r requirements.txt
```

> **注意**：ruff 可能需要单独安装（`pip install ruff`），不在 requirements.txt 中。

---

## 6. 数据准备

1. 将 URFD 视频放入 `data/sources/urfd/`
2. 将 OF-Syn tar 放入 `data/omnifall/data_files/omnifall-synthetic_av1.tar`
3. 生成 manifest：

```bash
python tools/build_manifest.py
python tools/prepare_omnifall_events.py
```

`data/manifest.csv` 约 12,100 行，包含 URFD 普通路径和 OF-Syn `tar://` URI。

---

## 7. 下一步执行路径

### Gate 3：NPZ → TCN window consumer 验证

**目标**：用 20 个 canary NPZ 实现最小 TCN window consumer 测试，验证 padding/mask/标签边界。

```python
# 在 models/ 或 tests/ 中实现：
# 1. 读取 NPZ → (T, P, 17, 3) dense array
# 2. 滑动窗口切分 [T_in, P, 17, 3] → [T_out]
# 3. 归一化：减去 torso center，除以 bbox 尺寸
# 4. padding/mask 正确传播
# 5. clip-level 标签正确分配
```

**验收**：在 canary 的 20 个 NPZ 上 `python -m pytest tests/test_tcn_window.py -q` 通过。

### Gate 4：全量 pose cache

```bash
# 完整 1,200 OF-Syn val cache (~59 min)
python -m tools.extract_keypoints \
  --manifest data/manifest.csv --dataset of-syn --split val \
  --cache-root runs/pose_cache_val --temp-root runs/tmp/pose_cache_val \
  --model yolo11n-pose.pt --device 0 --crop auto

# 10,800 OF-Syn train cache (~9.1h)
python -m tools.extract_keypoints \
  --manifest data/manifest.csv --dataset of-syn --split train \
  --cache-root runs/pose_cache_train --temp-root runs/tmp/pose_cache_train \
  --model yolo11n-pose.pt --device 0 --crop auto
```

完成后验证：

```bash
# 全量 cache resume
python -m tools.extract_keypoints \
  --manifest data/manifest.csv --dataset of-syn --split val \
  --cache-root runs/pose_cache_val --temp-root runs/tmp/pose_cache_val \
  --model yolo11n-pose.pt --device 0 --crop auto
# 预期：processed=0, resumed=1200, failed=0
```

### Gate 5：FallTCN 训练

```bash
# 使用只读 cache 训练（epoch 内零 YOLO）
python -m tools.train_tcn \
  --cache-root runs/pose_cache_train \
  --val-cache-root runs/pose_cache_val \
  --epochs 50 --batch-size 32
```

### Gate 6：val 完成后选择阈值

- rule-only, TCN-only, 融合 三组消融
- val P@R90/P@R95/MAP → 选择最优工作点

### Gate 7：test seal

- OF-Syn test 只消费一次
- URFD test 继续封存

### Gate 8：1080P/V100 pre-seal 硬门

- 在最终 test 前完成
- 1080P 输入端到端 P50/P95
- V100 TensorRT 结果

### Gate 9：匿名提交

- 清除身份、绝对路径、用户名、元数据
- PDF ≤200MB + MP4 ≤5min ≤200MB + ZIP ≤200MB
- 截止：2026-09-01 23:59

---

## 8. 依赖顺序图

```
[done] Gate 0: Round 0 确定性基线
  ↓
[done] Gate 1: 可复用推理引擎 + URFD val 批量评测
  ↓
[done] Gate 2A: 统一视频源 + pose cache + cache-first 提取
  ↓
[done] Gate 2B: rule evaluator tar 接入 + 20-clip canary
  ↓
[now] Gate 3: NPZ→TCN window consumer 验证
  ↓
[next] Gate 4: 全量 cache (train 10,800 + val 1,200)
  ↓
Gate 5: FallTCN 训练 (只读 cache, epoch 内零 YOLO)
  ↓
Gate 6: val 消融 + 阈值选择
  ↓
Gate 7: test seal (OF-Syn test 一次, URFD test 封存)
  ↓
Gate 8: 1080P/V100 pre-seal 硬门 (必须在 test 之前!)
  ↓
Gate 9: 匿名提交
```

---

## 9. 关键约束

| 约束 | 说明 |
|---|---|
| 截止 | 2026-09-01 23:59 |
| 测试 | test split 无 override；OF-Syn test 只消费一次 |
| 参数限制 | ≤20M 参数，FP32 ≤80MB（当前 ~12MB，有余量） |
| 时延 | ≤100ms（当前 RTX 4060 微基准 52.82ms P95） |
| C 盘 | 仅剩 ~5.8GB，禁止用于临时文件 |
| D 盘 | 剩余 ~83GB |
| 匿名 | 提交材料清除所有身份/路径/用户名 |
| 环境 | 最终提交需 V100 复测 |
| 官方协议 | event-level 匹配细节未完全确认；URFD 结果为本地 clip 代理 MAP |

---

## 10. 禁止事项

- 不读取 test（除非在 Gate 7 冻结条件下，由 seal 约束）；
- 不把 test 纳入阈值选择、模型选择、消融或增强调参；
- 不直接跑 12,000 clips（先 canary → 再全量）；
- 不在 C: 默认 Temp 解包 tar 视频；
- 不声称 OF-Syn subject-independent（上游只有随机 clip split）；
- 不把 RTX 4060、小分辨率时延冒充 V100/1080P 官方结果；
- 不在 train/val cache 完成前启动 TCN 训练；
- 不在 test seal 之前消耗 test 数据；
- 1080P/V100 硬门必须在 test 之前完成，不能先看 test 再决定是否合格。

---

## 11. 已知限制

- OF-Syn 完整 1,200 val 尚未缓存/评测（Gate 4）；
- dense `[T,P,...]` schema 已 smoke，但 TCN dataset consumer 尚未实现（Gate 3）；
- 多目标 tracker 无 ReID，复杂遮挡/交叉可能产生 ID 碎片；
- 当前规则基线不是最终竞赛模型；
- 官方 event 匹配、V100/1080P 和 NPU 模型体积口径仍待确认；
- `eval/benchmark.py` 的 P50/P95 是 YOLO predict 微基准，不是完整端到端。

---

## 12. 关键文件索引

| 文件 | 作用 |
|---|---|
| `pipeline/video_source.py` | 本地 MP4/tar 成员统一解析、内容哈希、临时生命周期 |
| `pipeline/pose_cache.py` | 原子、严格 schema 的 NPZ cache |
| `pipeline/pose_extractor.py` | 单模型逐帧多人姿态提取 |
| `pipeline/pose_track.py` | 多目标姿态跟踪 |
| `tools/extract_keypoints.py` | cache-first 姿态批量提取 CLI |
| `eval/evaluate_manifest.py` | 断点批量评测（支持 tar URI） |
| `eval/metrics.py` | clip/event 两种评测口径 |
| `models/tcn.py` | 0.134M 因果 FallTCN |
| `pipeline/inference_engine.py` | 单模型复用推理引擎 |
| `data/manifest.csv` | 约 12,100 行，URFD + OF-Syn 数据索引 |
| `.hermes/plans/2026-08-17_015002-competition-gap-and-execution-plan.md` | 权威两周计划 |
| `.hermes/plans/2026-08-17_144818-pose-cache-gate2.md` | Gate 2 详细计划 |

---

## 13. 验收 checklist（交接后）

- [ ] 在新机器 `python -m pytest -q` 通过（121 tests）
- [ ] `ruff check .` 通过
- [ ] 数据下载并 `build_manifest.py` + `prepare_omnifall_events.py` 生成 manifest
- [ ] 4-clip smoke 重跑：4/4 成功，第二次 4/4 resume
- [ ] 20-clip canary 重跑：20/20 成功，第二次 20/20 resume
- [ ] Gate 3: TCN window consumer 测试通过
- [ ] Gate 4: 全量 cache 完成并 resume 验证
- [ ] Gate 5: TCN 训练开始
