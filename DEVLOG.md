# 开发日志

> 本文件记录 Detictive 跌倒检测竞赛项目的关键开发决策和里程碑，供接手者参考。
> 详细技术交接见 `HANDOVER.md`，赛题要求见 `面向低算力端侧平台基于视觉的实时跌倒检测.docx`。

## 2026-08-17 — Gate 2 完成

### 背景

项目已完成 Round 0（确定性规则基线，74 tests）和 Round 1（可复用推理引擎 + URFD val 批量评测，86 tests，MAP=80.0%），进入 Gate 2：统一视频源与离线 pose cache。

### Gate 2A：统一视频源 + pose cache

**核心实现：**

- `pipeline/video_source.py`：本地 MP4 与 `tar://archive!/member` 统一解析；tar handle 批次复用；显式 D: temp root 临时文件管理。
- `pipeline/pose_cache.py`：schema v1，原子 NPZ 写入（flush/fsync/replace），`allow_pickle=False`，严格 dtype/shape/时间轴/padding 验证。
- `pipeline/pose_extractor.py`：单模型延迟加载，100% resume 时零 YOLO 构造；每 clip 重置 tracker；空检测帧保留。
- `tools/extract_keypoints.py`：cache-first manifest 驱动 CLI，partial failure 隔离，机器可读 telemetry。

**提交链：**

```
788a077 feat: add unified local and tar video source
76f5ef0 feat: add atomic validated pose cache schema
8e4539f feat: extract resumable pose caches from manifest
a860a2d feat: harden pose cache resume telemetry
bc3ad28 fix: enforce deterministic pose track rows
```

**首次 focused review 发现 4 个 P1：**

1. 提前 EOF：OpenCV 解码帧数少于容器报告值时，当前实现会发布截断 cache。
2. 本地视频 TOCTOU：源哈希与实际解码字节不是同一不可变快照。
3. 模型权重 TOCTOU：签名哈希与延迟加载 YOLO 可能消费不同字节。
4. BadZipFile：损坏 NPZ 的 `zipfile.BadZipFile` 未纳入 cache miss 恢复路径。

**修复方案：**

```
f5f1ea0 fix: bind pose caches to consumed bytes
```

- 本地视频 cache miss 时复制到显式 temp root，SHA-256 与 decoder 消费同一快照。
- 模型权重同样冻结为不可变快照，签名和 lazy factory 消费同一字节。
- 容器报告帧数核对实际解码帧数；`max_frames` 截帧语义正确处理。
- `BadZipFile` 统一转为 `ValueError`，批处理将其视为 cache miss 并重建。

**双路 closure review：** `passed=true, remaining_issues=[]`

**合并：** fast-forward 到 `master@a52f37d`，118 tests passed。

### Gate 2B：rule evaluator tar 接入

**问题：** Gate 1 的 `eval/evaluate_manifest.py` 对 tar URI 执行 `Path(video_path)`，无法处理 OF-Syn tar 成员。

**实现：**

- evaluator 注入 `VideoSourceResolver`，批次复用 tar handle。
- local 行保持原 `Path` 直通 engine；tar URI 经 resolver materialize。
- JSONL 增加 `source_kind` 和 `source_prepare_seconds`。
- evaluator run signature 纳入 `video_source.py`。

**提交：**

```
45127cd feat: evaluate tar-backed manifest videos
```

**reviewer A 发现 P1：** 非法 tar URI（如 `tar://missing-delimiter`）在 per-clip `try` 块外抛出，绕过失败隔离并可能卡死 test seal。

**修复：** 将 `_resolve_video_source()` 和 `parse_video_source()` 移入 per-clip `try`；`source_kind` 默认 `"unknown"`，解析成功后才更新。

```
e9496d7 fix: isolate bad tar URI as per-clip error
```

**closure review：** `passed=true, remaining_issues=[]`

**合并：** fast-forward 到 `master@e9496d7`，121 tests passed。

### 20-clip canary

**选样策略：**
- URFD 4 clips：2 fall + `adl-12-cam0`、`adl-40-cam0` 两个已知 hard negatives。
- OF-Syn 16 clips：按 `sha256(clip_id)` 在 val 正负类各取 8。
- Manifest SHA-256：`fdea91af...c45732df`

**首跑结果：**

| 数据集 | Clips | 时间 | Cache 大小 |
|---|---|---|---|
| URFD | 4 | 13.76s | 148KB |
| OF-Syn | 16 | 47.69s | 645KB |
| 合计 | 20 | 61.45s | 793KB |

**重跑：** 20/20 resumed，`processed=0`，零 YOLO 调用。

**外推：**

| 场景 | 预估时间 | 预估存储 |
|---|---|---|
| 1,200 OF-Syn val | ~59 min | ~47.6MB |
| 10,800 train | ~9.1h | ~429MB |

**决策：** D: 剩余 83GB，截止 8/31 剩 14 天，时延/存储余量 ≥30%。选择全量 train + 完整 val，无需冻结子集。

### 下一步路径

```
1. [x] Gate 2A: 统一视频源 + pose cache + cache-first 提取
2. [x] Gate 2B: rule evaluator tar URI + 20-clip canary
3. [ ] Gate 3: NPZ→TCN window consumer contract（用 canary NPZ 验证 padding/mask/标签边界）
4. [ ] Gate 4: 全量 10,800 train + 1,200 val cache（后台 ~10h）
5. [ ] Gate 5: FallTCN 训练（只读 cache，epoch 内零 YOLO）
6. [ ] Gate 6: val 消融 + 阈值选择
7. [ ] Gate 7: 1080P/V100 pre-seal 硬门 + test seal
8. [ ] Gate 8: 匿名提交（PDF + MP4 + ZIP，截止 2026-09-01 23:59）
```

### 关键设计决策

| 决策 | 理由 |
|---|---|
| tar URI 用 `tar://archive!/member` 格式 | 与本地路径有明确区分，parser 要求恰好一个 `!/` |
| 本地视频也复制到 temp root | 防止 ABA 替换导致 SHA 不匹配内容 |
| 模型权重快照复用 | 签名和 lazy factory 消费同一字节，100% resume 仍不构造 YOLO |
| NPZ 无 pickle | `allow_pickle=False` 消除任意代码执行风险 |
| 写入用 flush/fsync/replace | 原子性，失败不破坏旧 cache |
| `BadZipFile` 归一化为 `ValueError` | 统一进入 cache miss 重建路径 |
| Per-clip try 包含 source resolve | 非法 URI 不绕过失败隔离 |
| test split 无 override | CLI 直接拒绝，evaluator 需 seal |
| 显式 D: temp root | C: 仅剩 5.8GB，禁止用于临时文件 |
| tar handle 批次复用 | OF-Syn 9.7GB archive，不能反复打开 |
