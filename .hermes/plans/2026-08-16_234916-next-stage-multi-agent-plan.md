# 跌倒检测竞赛下一阶段多 Agent 实施计划

> **For Hermes:** 按“先冻结基线、再分 worktree 并行、最后串行集成”的顺序执行。所有代码任务遵循 TDD；每个 worktree 合并前必须独立审查。

**Goal:** 把当前“单样本可运行的规则 Demo”推进为“可在完整验证集量化、可训练 TCN、可复现消融、可生成比赛材料”的候选作品。

**Architecture:** 首先修正交接文档中的完成度口径，并把当前未提交基线作为一个可回滚 checkpoint。第一轮并行解决数据读取/关键点缓存、批量评测、红外增强三个互不冲突的问题；第二轮依赖关键点缓存训练 TCN；第三轮把 TCN 接入推理并在完整验证集上做规则/TCN/融合消融。评测定义、融合调参和最终提交材料由主线 Agent 负责，不外包。

**Tech Stack:** Python 3.11、PyTorch 2.13、Ultralytics 8.4、OpenCV 5、pytest、NumPy；Windows 10 + Git Bash；RTX 4060 Laptop 8GB。

---

## 0. Review 结论与完成度校准

### 已真实完成

- `python -m pytest -q`：**42 passed**。
- YOLO11n-pose + 单主目标跟踪 + 规则融合 + 事件聚合已在单个 URFD 跌倒视频实跑。
- `reports/demo_fall01_final.json` 检出约 3.63–3.80s 的候选事件。
- `eval/benchmark.py` 实测 YOLO11n-pose 2.874M 参数，RTX 4060 PyTorch eager P50 47.92ms、P95 52.82ms。
- `data/manifest.csv` 共 12,100 行：OF-Syn 12,000，URFD 100；当前 group split 检查没有跨 split 泄漏。
- `fall` 与 `fallen` 已分开保存，并共同映射到 `fall_incident`。

### 当前交接文档的过度表述

以下内容不能再报告为“完成”，应在下一轮修正 `HANDOVER.md`：

1. `HANDOVER.md:18` 把“本地评测集”标为完成，但目前只有 manifest，**没有完整 val/test 推理结果和 P@R90/P@R95**。
2. `HANDOVER.md:20` 把 YOLO11n-pose 规则 Demo 称作“通道 A 基线”；按架构定义它属于姿态通道 C，而不是外观通道 A。
3. `HANDOVER.md:22` 只有 1 个 fall + 1 个 ADL 的烟雾验证，不能等同于完整 D4 出口标准。
4. `HANDOVER.md:232-234` 给出 TCN 训练集数量和 AUC/F1 目标，但目前没有关键点缓存、窗口 Dataset、训练脚本或真实基线，目标只能作为期望值，不能作为已验证事实。

### 新发现的关键缺口

- `tools/build_manifest.py:150` 的 OF-Syn URI 使用 `...!/fall/...mp4`，而 tar 成员实际带 `./fall/...mp4`；必须由统一 TarVideoReader 规范化，不能直接把 URI 交给 OpenCV。
- `infer.py:129` 每次 `analyze_video()` 都重新加载 YOLO；`infer.py:120-127` 强制创建 MP4。完整数据集批量评测会非常低效，需要可复用 engine 和 `render=False`。
- URFD manifest 只有 clip 标签，`events_json=[]`；URFD 只能先做 clip-level 指标，不能伪造 event-level GT。
- 当前只跟踪一个主目标；多人片段可能漏掉非主目标跌倒。第一版 TCN 可先保持单目标，但必须在报告中声明，并在批量 FP/FN 分析中统计多人失败。
- `eval/benchmark.py` 只测 YOLO predict，不是 `decode + crop + pose + tracking + rules/TCN + aggregation` 的完整端到端时延。
- OF-Syn 数据仍在 tar 中，尚无可随机读取/缓存的训练数据管线。

---

## 1. 总体执行策略

### 为什么不能马上开 3 个 worktree

当前 `master` 有 6 个修改文件和 17 个未跟踪文件。Git worktree 只能基于已提交内容，新的 worktree 看不到这些改动。因此必须先完成 **Round 0：冻结基线 checkpoint**。

### 推荐轮次

| 轮次 | 并行度 | 内容 | 依赖 |
|---|---:|---|---|
| Round 0 | 1（主线） | 修正现有 P0/P1、更新交接、验证并 commit | 无 |
| Round 1 | 3 worktrees | A 数据/关键点缓存；B 批量评测；C 红外增强 | Round 0 commit |
| Round 2 | 2 worktrees | D TCN 训练；E 基线误报分析工具 | A、B 合并 |
| Round 3 | 2（先并行后串行） | F TCN 推理接入；G 消融/阈值搜索；主线最终融合 | D、E 合并 |
| Round 4 | 2 agents | 文档初稿 + 视频脚本；主线做合规终审 | 指标冻结 |

### 主线必须亲自负责

- 官方评测协议确认和 `eval/metrics.py` 最终定义。
- 融合权重、阈值搜索和最终模型选择。
- 所有 agent 产物的测试、文件回读和指标复核。
- 提交文档中的数字、匿名性、命名和最终打包。

---

## 2. Round 0 — 冻结并提交当前端到端基线

### Task 0.1：按 focused review 修正现有基线

**Objective:** 只修复当前 diff 中确认的 P0/P1，不加入下一阶段新功能。

**Files:**
- Modify: `HANDOVER.md`
- Potentially modify after review: `eval/metrics.py`, `tools/build_manifest.py`, `infer.py`
- Tests: existing `tests/`

**Steps:**

1. 汇总三个 focused reviewer：metrics、data/leakage、pipeline。
2. 每个 P0/P1 先写/补失败测试，再做最小修复。
3. 将 `HANDOVER.md` 中“完整评测已完成”的表述改为“单样本规则基线已跑通”。
4. 更新 `docs/research/source_downloads.md`：URFD 和 OF-Syn 已完成下载与校验，Le2i/UP-Fall 暂缓。
5. 执行：
   ```bash
   python -m pytest -q
   git diff --check
   ```
   预期：42+ tests passed；无 whitespace error。
6. 执行静态安全扫描，确认无硬编码密钥、`shell=True`、`eval/exec`、pickle 反序列化。
7. 做一个不写视频的端到端 smoke（若该选项尚未实现，Round 1B 再补；本轮保留现有实跑证据即可）。
8. 提交 checkpoint：
   ```bash
   git add -A
   git commit -m "feat: establish leak-aware end-to-end fall baseline"
   ```

**Exit criteria:** `master` 工作区干净；42+ tests pass；当前所有新代码都进入 Git；`HANDOVER.md` 无夸大完成度。

---

## 3. Round 1 — 三 worktree 并行

> 创建 worktree 前先记录 Round 0 commit SHA。三个分支只创建各自新文件，尽量避免同时修改 `infer.py`、`README.md`。

### 3.1 Worktree A：Tar 数据读取与关键点缓存

**Branch:** `feat/pose-cache`  
**Worktree:** `D:\HermesWorkspace\Detictive-wt-pose-cache`

```bash
git worktree add ../Detictive-wt-pose-cache -b feat/pose-cache
```

**Objective:** 让 OF-Syn tar 和普通 MP4 通过同一接口可读，并将视频离线转成可重复使用的关键点缓存。

**Files:**
- Create: `pipeline/video_source.py`
- Create: `tools/extract_keypoints.py`
- Create: `tests/test_video_source.py`
- Create: `tests/test_extract_keypoints.py`
- Create: `docs/data_schema.md`

**接口先冻结：**

每个视频输出一个 `.npz`：

- `keypoints`: float32 `(N, 17, 3)`，bbox 内归一化坐标与 confidence。
- `boxes`: float32 `(N, 4)`，原图归一化 xyxy。
- `timestamps`: float32 `(N,)`。
- `detected`: uint8 `(N,)`。
- `fps`, `width`, `height`, `clip_id`, `dataset`, `split`: metadata。
- 索引 CSV：`clip_id,dataset,split,cache_path,frame_count,fps,status,error`。

**TDD tasks:**

1. 测试普通 MP4 reader 返回连续帧和正确 FPS。
2. 测试 tar URI 同时接受 `member` 与 `./member`，并能解码 OF-Syn 样例。
3. 测试缓存 schema、dtype、shape 和 metadata。
4. 测试中断恢复：已存在且校验通过的 `.npz` 跳过；坏缓存重做。
5. 提取 4 个样例（OF-Syn fall/non-fall、URFD fall/ADL）并回读验证。
6. 先跑 50 个 OF-Syn train + 全部 URFD val/test 的 smoke；不要直接启动 12,000 视频全量任务。

**Verification:**

```bash
python -m pytest tests/test_video_source.py tests/test_extract_keypoints.py -q
python tools/extract_keypoints.py --manifest data/manifest.csv --split val --limit 20 --output data/pose_cache
```

**Agent prompt:**

```text
你负责 feat/pose-cache worktree。严格 TDD，不得修改 infer.py、eval/metrics.py、pipeline/fusion.py。先实现 pipeline/video_source.py，统一读取普通 MP4 和 tar://...!/member.mp4；OF-Syn tar 成员实际可能带 './'，必须规范化。再实现可断点恢复的 tools/extract_keypoints.py，输出 docs/data_schema.md 中约定的 npz schema。控制在本轮代码范围内，不训练模型。每完成一个测试立即运行；最终返回测试结果、4个可回读缓存绝对路径、失败列表。
```

### 3.2 Worktree B：可复用推理引擎与完整 URFD 批量评测

**Branch:** `feat/batch-eval`  
**Worktree:** `D:\HermesWorkspace\Detictive-wt-batch-eval`

```bash
git worktree add ../Detictive-wt-batch-eval -b feat/batch-eval
```

**Objective:** 把单视频 Demo 改为一次加载模型、可关闭渲染的推理引擎，并在完整 URFD val/test 上得到第一份真实 clip-level P@R90/P@R95。

**Files:**
- Create: `pipeline/inference_engine.py`
- Create: `eval/evaluate_manifest.py`
- Create: `tests/test_inference_engine.py`
- Create: `tests/test_evaluate_manifest.py`
- Modify minimally: `infer.py`（只变为 CLI wrapper）
- Output: `reports/urfd_rule_baseline.json`, `reports/urfd_rule_baseline.md`

**TDD tasks:**

1. 测试 engine 只初始化一次模型，多视频复用。
2. 测试 `render=False` 时不创建 VideoWriter；`render=True` 保持现有 Demo 行为。
3. 测试每个 clip 输出 `clip_score`、events、processed_frames、失败原因。
4. 测试 batch evaluator 只按 manifest split 读取，未知/失败 clip 不能静默跳过。
5. 测试 URFD 使用 clip-level GT；`events_json=[]` 时禁止调用 event-level 指标。
6. 在 URFD val（14 个视频）跑完规则基线，再跑 test（16 个视频）。
7. 报告 P@R90、P@R95、MAP%、混淆工作点、失败视频和 P50/P95 端到端时延。

**Verification:**

```bash
python -m pytest tests/test_inference_engine.py tests/test_evaluate_manifest.py -q
python eval/evaluate_manifest.py --manifest data/manifest.csv --dataset urfd --split val --render false
```

**Agent prompt:**

```text
你负责 feat/batch-eval worktree。严格 TDD。把 infer.py 的核心推理抽到 pipeline/inference_engine.py，使 YOLO 只加载一次，并支持 render=False。实现 eval/evaluate_manifest.py，在 URFD val/test 上用 clip-level GT 调用现有 competition_map(mode='clip')。URFD 没有 event GT，绝不能计算 event MAP。不得修改 tools/build_manifest.py 或 models/tcn.py。最终必须真实跑完 URFD val，返回 JSON/MD 报告路径、P@R90/P@R95/MAP、失败视频和端到端 P50/P95。
```

### 3.3 Worktree C：红外/低光增强纯函数库

**Branch:** `feat/ir-augment`  
**Worktree:** `D:\HermesWorkspace\Detictive-wt-ir-augment`

```bash
git worktree add ../Detictive-wt-ir-augment -b feat/ir-augment
```

**Objective:** 实现可复现、可配置、可视化验证的红外/低光增强，但暂不接入训练主流程。

**Files:**
- Create: `tools/augment.py`
- Create: `tests/test_augment.py`
- Create: `configs/augment_ir.yaml`
- Create: `tools/visualize_augmentations.py`
- Output: `reports/augmentation_grid.jpg`（不提交大图）

**增强最小集：**

- RGB → 灰度三通道。
- gamma 低光。
- CLAHE（可选）。
- 高斯/泊松传感器噪声。
- 轻微模糊。
- CutOut 遮挡。

**TDD tasks:**

1. 每个增强保持 `uint8 H×W×3`。
2. 固定 seed 时输出完全一致。
3. 灰度三通道的三个 channel 相等。
4. 参数越界明确报错。
5. 空概率/identity 不改变图像。
6. 生成 12 格可视化网格人工抽查。

**Agent prompt:**

```text
你负责 feat/ir-augment worktree。严格 TDD，只创建 tools/augment.py、tests/test_augment.py、configs/augment_ir.yaml、tools/visualize_augmentations.py，不修改训练、推理或评测文件。所有增强保持 uint8 HxWx3，可注入 numpy RNG，固定 seed 必须可复现。实现灰度三通道、gamma低光、CLAHE、噪声、模糊、CutOut。最终运行测试并生成 reports/augmentation_grid.jpg，报告绝对路径供主线人工查看。
```

### Round 1 合并顺序

1. `feat/pose-cache`（定义数据 schema）。
2. `feat/batch-eval`（锁定规则基线真实指标）。
3. `feat/ir-augment`（独立、冲突最少）。

每个分支合并前：

```bash
python -m pytest -q
git diff master...HEAD --check
```

主线回读产物并重新运行测试，不采信 Agent 自报。

---

## 4. Round 2 — TCN 数据集、训练与误报分析

### 4.1 Worktree D：TCN Dataset 与训练

**Branch:** `feat/tcn-training`

**Objective:** 基于 pose cache 训练第一版 FallTCN，保存 checkpoint、配置和可复现实验报告。

**Files:**
- Create: `data/pose_windows/`（gitignore）
- Create: `models/pose_dataset.py`
- Create: `scripts/train_tcn.py`
- Create: `eval/evaluate_tcn.py`
- Create: `configs/tcn_baseline.yaml`
- Create: `tests/test_pose_dataset.py`
- Create: `tests/test_train_tcn.py`
- Output: `runs/temporal/tcn_baseline/best.pt`, `reports/tcn_baseline.md`

**关键标签策略：**

- 首版窗口目标以“窗口末帧是否处于 `fall_process` 或 `post_fall_state`”定义。
- 同时保留 `event_semantics`，后续对 `fall_process` 与 `post_fall_state` 分别报告召回。
- hard negatives 要分层采样：sit/lie/kneel/squat/crawl 不得被普通 background 淹没。
- 缺失关键点使用 `detected=0` 和 confidence=0 表示；禁止用未来帧插值进入因果模型。
- split 只来自 manifest，训练脚本禁止重新随机划分。

**训练验收：**

- 小数据 overfit 测试：32 个窗口上 loss 明显下降。
- 固定 seed 可复现 split 与首轮 loss。
- checkpoint 包含 model/config/normalization/label_mapping/seed。
- 至少报告 val ROC-AUC、PR-AUC、F1、P@R90、P@R95；不得只报告 accuracy。
- 先做 1 epoch smoke，再后台运行完整训练：
  ```bash
  python scripts/train_tcn.py --config configs/tcn_baseline.yaml
  ```

**Agent prompt:**

```text
你负责 feat/tcn-training。输入只允许使用已合并的 data/pose_cache 索引和 data/manifest.csv，不重新跑 YOLO。严格 TDD。实现因果窗口 Dataset、分层负采样、训练/验证、checkpoint 和评测。禁止未来帧插值；缺失点用 detected/confidence 表达。先让 32 个窗口 overfit，再跑 1 epoch smoke，长训练用后台进程。最终返回 checkpoint 绝对路径、配置、完整指标和训练日志，不要修改 eval/metrics.py 或 infer.py。
```

### 4.2 Worktree E：FP/FN 自动导出与归因

**Branch:** `feat/error-analysis`

**Objective:** 把 Round 1 的完整 URFD 预测转为可操作的误报/漏报清单。

**Files:**
- Create: `eval/error_analysis.py`
- Create: `tests/test_error_analysis.py`
- Output: `reports/fp_analysis.md`, `reports/error_clips/*.jpg`

**内容：**

- 在 R90 和 R95 两个工作点分别列 FP/FN。
- 导出关键帧 contact sheet，不复制整段大视频。
- 自动统计：无人体检测率、低关键点覆盖率、track switch 次数、最大下坠、最小 verticality、多人数量。
- 归因标签：pose miss / track switch / perspective / lying confusion / threshold / unknown。

---

## 5. Round 3 — TCN 接入、完整消融与阈值搜索

### Task 5.1：TCN 推理接入

**Files:**
- Modify: `pipeline/inference_engine.py`
- Modify: `pipeline/fusion.py`
- Create/modify: `tests/test_tcn_inference.py`, `tests/test_fusion.py`

**Requirements:**

- 规则 only、TCN only、规则+TCN 三种模式配置化。
- TCN 只消费过去与当前 16 帧，禁止未来信息。
- checkpoint schema/输入 normalization 不匹配时 fail fast。
- 多人限制继续显式记录；不在本阶段贸然重构多目标 TCN。

### Task 5.2：消融与阈值搜索

**Files:**
- Create: `eval/run_ablation.py`
- Create: `configs/fusion_grid.yaml`
- Output: `reports/ablation.md`, `reports/threshold_search.json`

**实验矩阵：**

1. 规则 only。
2. TCN only。
3. 规则 + TCN。
4. 输入尺寸 416 / 512 / 640（只在胜出融合方案上测）。
5. RGB / 灰度 / 低光 / 遮挡子集。

**Selection rule:** 只用 val 选阈值和权重；test 只在冻结后运行一次，禁止用 test 调参。

**Final gate:**

- 生成真实 P@R90/P@R95/MAP 表。
- 参数量按全部推理模型求和。
- 时延报告为完整端到端 P50/P95，而不是仅 YOLO predict。
- 若 TCN 没有稳定提升 MAP，保留规则基线，不因“方案复杂”强行加入。
- 只有在消融证明外观通道有增益时，才启动第二个 YOLO 模型。

---

## 6. Round 4 — 提交材料并行

### Agent M1：项目文档初稿

输入必须是冻结后的：

- `reports/urfd_rule_baseline.md`
- `reports/tcn_baseline.md`
- `reports/ablation.md`
- `reports/benchmark*.json`
- 赛题 DOCX

输出 `deliverables/project_document_draft.md`，所有数字附来源文件路径；禁止填编造指标。

### Agent M2：5 分钟视频脚本与分镜

输出 `deliverables/video_storyboard.md`：痛点 30s → 架构 45s → RGB/低光/红外演示 150s → 指标/轻量化 45s → 总结 30s。

### 主线终审

- 匿名性：不得出现学校、导师、个人身份。
- 简介 ≤300 字。
- 视频 ≤5min、≤200MB、MP4。
- ZIP ≤200MB，并在干净环境按 README 重跑。
- 所有指标与报告 JSON 反向核对。

---

## 7. 多 Agent / Worktree 运行纪律

1. 子 Agent 固定 600 秒，任务必须限定 ≤10 次工具调用；复杂代码任务拆小，不派“全仓审查”。
2. 下载和长训练不用 `delegate_task`；主线用 `terminal(background=true, notify_on_complete=true)`。
3. 每个 worktree 的 agent 只修改明确文件集合，禁止 drive-by refactor。
4. 子 Agent 完成后主线必须：读文件、跑测试、检查产物、核对指标。
5. 每个分支一次独立 focused review：data / metrics / pipeline 分开审，不再派全仓 reviewer。
6. `data/`, `runs/`, 大视频和中间缓存不提交；提交 schema、脚本、配置和小型 JSON/MD 报告。
7. 不在多个 worktree 同时占用同一块 GPU。GPU 队列由主线串行调度：pose cache → TCN train → ablation。

---

## 8. 本轮立即决策

推荐执行顺序：

1. **本轮只完成 Round 0**：消化 focused reviews、修正 P0/P1、更新交接、commit。
2. 下一轮创建三个 worktree，启动 Round 1。
3. Round 1 结束后用完整 URFD val 指标决定是否调整规则，再启动 TCN 训练。

不要在当前未提交状态直接开 worktree，也不要在没有完整规则基线指标前直接训练 12,000 视频的 TCN。
