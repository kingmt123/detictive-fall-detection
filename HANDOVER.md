# 视觉实时跌倒检测竞赛项目 — AI 交接文档

> **更新时间**：2026-08-17
> **当前分支**：`feat/pose-cache`（worktree：`D:\HermesWorkspace\Detictive-wt-pose-cache`）
> **基线分支**：`master` @ `1624ec0`
> **状态**：Gate 1 已合并；Gate 2A 的 4 个 review P1、118 tests、最终 4/4 rebuild→resume 和双路 closure review 均通过，等待 fast-forward 合并
> **下一门**：合并 Gate 2A；再补 evaluator/InferenceEngine tar-source tracer，运行冻结的 20-clip val canary并验证最小 NPZ→TCN consumer；不读取 test

## 1. 当前事实

| 项目 | 当前状态 | 证据 |
|---|---|---|
| URFD val 规则基线 | 14/14 完成 | P@R90=0.80、P@R95=0.80、本地 clip MAP=80.0% |
| Test split | 未使用 | 批量评测有持久化 one-shot seal；pose cache 阶段直接拒绝 test |
| 普通 MP4 | 支持 | `pipeline/video_source.py` |
| OF-Syn tar URI | Pose cache 路径支持 | 严格 `tar://archive!/member`，批次内复用 tar handle；rule evaluator 尚待接入 |
| Pose cache | 支持 | 原子 NPZ、严格 dtype/shape、源内容 SHA、提取签名 |
| Cache-first 提取 | 支持 | `tools/extract_keypoints.py`，单模型复用、显式 dataset/split |
| 真实 smoke | 4/4 完成 | 2 URFD val + 2 OF-Syn val；重跑 4/4 resume |
| 自动化测试 | 118 passed | `python -m pytest -q` |
| FallTCN | 仅结构/单测 | 尚无正式 train cache、checkpoint、真实指标 |

## 2. Gate 2 实现

### 2.1 统一视频源

`pipeline/video_source.py`：

- cache miss 时本地 MP4 复制到显式 D: temp root 的不可变快照，hash 与 decoder 消费同一份字节；
- tar 成员流式写入调用方指定的 D: 临时目录；
- 一个 `VideoSourceResolver` 在整个批次复用 tar 索引/handle；
- 正常退出和异常退出都删除临时视频；
- resume 先检查廉价源身份，再流式检查完整内容 SHA-256；tar resume 不落临时文件。

不要改成系统默认 Temp。当前 C: 仅剩约 5.8GB，而 D: 尚有约 83GB。

### 2.2 Pose cache schema v1

每个 clip 一个 NPZ，`allow_pickle=False`：

- `frame_indices`: `int64 [T]`，必须从 0 连续；
- `timestamps`: `float64 [T]`，必须等于 `frame_indices/fps`；
- `keypoints`: `float32 [T,P,17,3]`；
- `bboxes`: `float32 [T,P,4]`；
- `track_ids`: `int64 [T,P]`；
- `valid_mask`: `bool [T,P]`；
- padding 的 `track_id=-1`；
- `frame_size=[height,width]`；
- 元数据包含源身份、源内容 SHA-256、模型/源码/依赖/参数签名。

写入流程：同目录临时文件 → flush/fsync → 自校验 → `os.replace`。旧 cache 只有在新 cache 完整写成后才被替换。

### 2.3 提取签名

`PoseExtractor.cache_signature()` 绑定：

- 模型权重 SHA-256；
- 签名和懒加载构造器消费同一个不可变权重快照；
- `pose_extractor.py`、`pose_track.py` 和实际裁剪实现源码；
- NumPy/OpenCV/PyTorch/Ultralytics 版本；
- device、image size、confidence、crop；
- `max_frames`，防止 smoke 截帧 cache 被全片训练误复用。

## 3. 真实 smoke 证据

使用 val-only 样本：

| Dataset | Clip | 标签 | 帧数 | 最大同时观测人数 P | 有效 observations | FPS | cache frame size |
|---|---|---:|---:|---:|---:|---:|---|
| URFD | `fall-18-cam0` | 1 | 65 | 5 | 74 | 30 | 240×320 |
| URFD | `adl-02-cam0` | 0 | 180 | 1 | 171 | 30 | 240×320 |
| OF-Syn | `fall/fall_ch_026` | 1 | 81 | 4 | 95 | 16 | 720×1280 |
| OF-Syn | `fall/fall_ch_085` | 0 | 81 | 3 | 91 | 16 | 720×1280 |

验收：

- 首次：两组均 `processed=2, resumed=0`；
- 第二次：两组均 `processed=0, resumed=2`，不调用 YOLO；
- 四个 NPZ 的 `T` 与重新逐帧解码数完全一致；
- 四个源内容 SHA-256 与 cache 一致；
- tar 临时目录退出后为空；
- `signature_max_frames=null`，本次均为全片 cache；
- test split 未读取。

smoke cache 位于 feature worktree 的 `runs/pose_cache_smoke/`，被 Git 忽略，不是提交资产。

## 4. 复现命令

```bash
cd /d/HermesWorkspace/Detictive-wt-pose-cache

python -m tools.extract_keypoints \
  --manifest D:/HermesWorkspace/Detictive/data/manifest.csv \
  --dataset urfd --split val \
  --cache-root runs/pose_cache_smoke \
  --temp-root runs/tmp/pose_cache_smoke \
  --model D:/HermesWorkspace/Detictive/yolo11n-pose.pt \
  --device 0 --crop auto \
  --clip-id fall-18-cam0 --clip-id adl-02-cam0

python -m tools.extract_keypoints \
  --manifest D:/HermesWorkspace/Detictive/data/manifest.csv \
  --dataset of-syn --split val \
  --cache-root runs/pose_cache_smoke \
  --temp-root runs/tmp/pose_cache_smoke \
  --model D:/HermesWorkspace/Detictive/yolo11n-pose.pt \
  --device 0 --crop auto \
  --clip-id fall/fall_ch_026 --clip-id fall/fall_ch_085
```

## 5. Git 状态和提交

Feature 分支提交：

```text
788a077 feat: add unified local and tar video source
76f5ef0 feat: add atomic validated pose cache schema
8e4539f feat: extract resumable pose caches from manifest
a860a2d feat: harden pose cache resume telemetry
bc3ad28 fix: enforce deterministic pose track rows
f5f1ea0 fix: bind pose caches to consumed bytes
```

计划 checkpoint 已在 master：

```text
1624ec0 docs: define pose cache gate and acceptance criteria
```

在 focused review 和最终回归通过前不要合并；不要 push，除非用户明确要求。

## 6. 下一步：20-clip canary

权威计划：

- `.hermes/plans/2026-08-17_015002-competition-gap-and-execution-plan.md`
- `.hermes/plans/2026-08-17_144818-pose-cache-gate2.md`

执行顺序：

1. fast-forward 合并已通过双路 closure review 的 Gate 2A；
2. TDD 接入 `evaluate_manifest -> InferenceEngine -> VideoSourceResolver` 的 tar URI，规则/event 逻辑不变；
3. 从 val 确定性选择 20 clips，覆盖 URFD/OF-Syn、正负样本和 hard negatives；
4. 记录每 clip：源读取/解包时间、YOLO+tracking 时间、总时间、NPZ 大小、T/P/有效 observation、失败类型；
5. 立即重跑，必须 20/20 resume 且零 YOLO；
6. 用 canary NPZ 先验证最小 per-track window、padding/mask 和标签边界 consumer；
7. 根据实测吞吐外推 1,200 val 和 train cache 成本；
8. 只有 canary 无 schema/恢复问题后才冻结配置并扩容；
9. 只有正式 train/val cache 覆盖率和反向审计通过后才启动 TCN pilot。

## 7. 禁止事项

- 不读取 test；pose cache CLI 没有 test override；
- 不把 test 纳入阈值选择、模型选择、消融或增强调参；
- 不直接全跑 12,000 clips；
- 不把 `max_frames` smoke cache 混入正式 cache root；
- 不在 C: 默认 Temp 解包 tar 视频；
- 不声称 OF-Syn subject-independent；上游只有随机 clip split；
- 不把 RTX 4060、小分辨率本地时延冒充 V100/1080P 官方结果；
- 不在 Gate 2 证据不足时提前训练 TCN 或叠加外观通道。

## 8. 当前已知限制

- OF-Syn 完整 1,200 val 尚未缓存/评测；
- `eval/evaluate_manifest.py` / `InferenceEngine.analyze()` 尚未通过共享 resolver 消费 tar URI；
- 还没有 20-clip 吞吐和 cache 容量外推；
- dense `[T,P,...]` schema 已 smoke，但 TCN dataset consumer 尚未实现；
- 多目标 tracker 无 ReID，复杂遮挡/交叉可能产生 ID 碎片；
- 当前规则基线不是最终竞赛模型；
- 官方事件匹配、V100/1080P 和 NPU 模型体积口径仍待确认。
