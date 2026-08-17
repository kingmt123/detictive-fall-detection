# 视觉实时跌倒检测竞赛项目 — AI 交接文档

> **生成时间**: 2026-08-16 18:10  
> **项目状态**: Round 1 批量评测基础设施完成，86 测试全通过，URFD val 14/14 已实跑
> **下一步**: 统一普通视频/tar 视频读取并生成可断点 pose cache；仍不直接冒进训练

---

## 一、当前进度总览

### 1.1 已完成的里程碑

| 阶段 | 状态 | 产出 |
|---|---|---|
| **D1-1** 环境确认 | ✅ 完成 | RTX 4060 Laptop 8GB, CUDA 13.2, Python 3.11, PyTorch 2.13.0+cu126 |
| **D1-2** 数据盘点 | ✅ 完成 | 3 份调研报告: dataset_survey.md, sota_survey.md, lightweight_options.md |
| **D1-3** 统一标注格式 | ✅ 完成 | `data/annotations/events.csv` (71,838 合法片段, 17,800 跌倒状态段；过滤 8 个无效区间) |
| **D2-1** 本地评测数据 | 🟡 val 基线完成 | URFD val 14/14：P@R90=0.80、P@R95=0.80、本地 clip MAP=80.0%；test 未使用，OF-Syn val 待跑 |
| **D2-2** 评测脚本先行 | ✅ 完成 | `eval/metrics.py`、`eval/evaluate_manifest.py`；显式 split、单模型复用、断点缓存 |
| **D3-1** 姿态/规则 smoke | 🟡 原型可跑 | YOLO11n-pose 2.874M 参数；属于姿态+规则通道，不是外观通道 A |
| **D3-2** 事件聚合器 v1 | ✅ 完成 | `pipeline/event_aggregator.py` (16 测试) |
| **D4-1** 端到端 smoke | 🟡 单样本通过 | 多目标 `infer.py` 实跑 fall-01 检出 track 3 事件 (3.70-3.80s)，adl-01 为 0 事件 |
| **D4-2** 参数量/微基准 | ✅ 完成 | `reports/benchmark_rtx4060.json`；47.92ms 仅为 YOLO predict 微基准，非端到端时延 |
| **D4-3** 无渲染批量时延 | 🟡 本机小分辨率 | URFD val 每 clip 帧 P50 中位数 16.75ms、P95 中位数 21.97ms；非 1080P/V100 |

### 1.2 Round 0 checkpoint 前的 Git 历史 (8 commits)

```
f63c326 feat(tools): OmniFall labels -> unified event annotation CSV (52k segments)
c5ce6bd docs: source acquisition status; HF candidate datasets; lesson: downloads via mainline bg terminal
456a20d feat(models): causal FallTCN (~0.15M) with shape/param/causality/latency tests
c82a71a docs: dataset & sota surveys; data: omnifall labels/splits (annotation layer); note: videos sourced separately
cffcd7e feat(pipeline): event aggregator (smoothing + hysteresis + merge + min-dur) with tests
4d91201 docs: adopt YOLO11 + MGD distillation per lightweight research; commit research report
763d6f0 feat(eval): clip-level event metrics (TP/FP/FN, P@R90, P@R95, MAP) with toy tests
a7eb66d chore: project skeleton for fall detection competition
```

### 1.3 未提交的改动 (待 commit)

**修改文件 (6 个)**:
- `.gitignore` — 添加 `reports/*.mp4`, `reports/*.jpg`, `reports/*_debug.json`
- `.hermes/plans/2026-08-16_fall-detection-competition-plan.md` — 修正 YOLOv8 → YOLO11
- `eval/metrics.py` — clip/event 双模式；显式协议；确定性最大匹配；严格输入校验
- `tests/test_metrics.py` — 覆盖同分顺序、缺失预测、非法输入和百分制
- `tests/test_tcn.py` — 时延测试改为 smoke test (不阻断单元测试)
- `tools/prepare_omnifall_events.py` — 修正 fall/fallen 语义，并过滤零/负时长区间

**新增文件 (16 个)**:
- `README.md` — 项目文档
- `requirements.txt` — 当前验证环境的依赖版本清单
- `scripts/reproduce.sh` — 一键复现脚本
- `eval/benchmark.py` — 模型参数与 warm-up 后时延基准
- `infer.py` — 端到端视频推理 (视频 → 事件 JSON + 可视化 MP4)
- `pipeline/fusion.py` — 时序规则融合 (累计下坠 + 姿态变化)
- `pipeline/pose_track.py` — 多目标 IoU 轨迹生命周期管理；保留旧单目标类做兼容
- `pipeline/rules.py` — 姿态物理特征与规则分数
- `tools/__init__.py` — 工具包初始化
- `tools/build_manifest.py` — 无泄漏视频 manifest 构建
- `tests/test_data_manifest.py` — 数据标签语义与无泄漏划分测试
- `tests/test_fusion.py` — 时序规则融合测试
- `tests/test_pose_track.py` — 主目标姿态跟踪测试
- `tests/test_rules.py` — 纯函数姿态特征与物理规则测试

**报告文件 (不提交)**:
- `reports/benchmark_rtx4060.json` — RTX 4060 时延基准
- `reports/demo_fall01_final.json` — URFD fall-01 事件检测结果
- `reports/demo_fall01_final.mp4` — URFD fall-01 可视化 (带骨架/分数/告警横幅)
- `reports/demo_adl01_v2.mp4` — URFD adl-01 可视化 (无误报事件)

---

## 二、技术方案核心

### 2.1 架构设计

```
1080P 视频流
   │  (抽帧/降采样至算法输入 640×640，帧率 15~25fps)
   ▼
┌─────────────────────────────────────────────┐
│ 通道B: YOLO11-pose 17关键点                │──→ 骨架序列 K(t)
│         │                                    │
│         ▼                                    │
│   多目标跟踪 (独立 track_id 与生命周期)      │──→ 多条连续轨迹
│         │                                    │
│         ▼                                    │
│   时序规则融合 (累计下坠 + 姿态物理规则)      │──→ 逐帧分数 s(t)
│         │                                    │
│         ▼                                    │
│   事件聚合器 (平滑 + 滞回 + 合并 + 过滤)     │──→ 跌倒事件
└─────────────────────────────────────────────┘
```

### 2.2 关键技术点

1. **fall/fallen 语义修正**:
   - `fall` (label 1) = 跌倒动态过程 → `fall_process`
   - `fallen` (label 2) = 跌倒后倒地状态 → `post_fall_state`
   - 两者都属于 `fall_incident` (跌倒事故)，但分开保存
   - `hard_negative` = sit/lie/kneel/squat/crawl (易混淆但非真实跌倒)

2. **无泄漏数据划分**:
   - 按 `group_id` (subject/trial) 划分，同一试次的多机位只进同一集合
   - URFD: 30 跌倒 × 双机位 + 40 ADL = 100 MP4
   - OF-Syn: 12,000 合成视频 (9.7GB tar)
   - manifest 保留 `subject/trial/camera/split` 完整信息

3. **端到端推理**:
   - 输入: 普通 RGB / 灰度复制三通道 / URFD `depth|RGB` 横向拼接视频
   - 输出: 事件 JSON (`track_id`, `t_start`, `t_end`, `score`) + 带骨架/分数/告警横幅的 MP4
   - 自动裁剪: 识别 URFD 的横向拼接帧 (宽高比 > 2.2 时裁右半)

4. **时序规则融合**:
   - 累计下坠: 1.2 秒窗口内的净向下位移 (质心 y 坐标)
   - 姿态变化: 躯干由竖直转为倾斜 (verticality < 0.8)
   - 两者相乘: 只有"快速下坠 + 倾斜"共同出现才给高分，抑制"走向摄像机"误报
   - 每条人物轨迹独立保存 scorer、质心历史和不可观测状态；短暂漏检不再写成明确 0 分
   - 事件聚合采用因果平滑，事件结束于最后一个有效高分帧，短尖峰先过滤再合并

5. **评测协议**:
   - **event 模式** (本地代理): temporal IoU ≥ 0.3, 确定性最大基数一对一匹配
   - **clip 模式** (官方若采用): 整段视频二分类
   - 两种模式都返回 `map` (0-1) 和 `map_percent` (0-100)
   - **注意**: 官方尚未明确事件匹配协议，event 模式是显式假设

### 2.3 性能指标 (实测)

| 指标 | 数值 | 说明 |
|---|---|---|
| YOLO11n-pose 参数量 | **2.874M** | 远低于 20M 约束 |
| fp32 权重体积估算 | **11.50MB** | 远低于 80MB 约束 |
| RTX 4060 P50 时延 | **47.92ms** | PyTorch eager, 含 Python 前后处理 |
| RTX 4060 P95 时延 | **52.82ms** | 同上 |
| TCN 参数量 | **0.134M** | 约束 ≤0.5M |
| TCN CUDA 推理 | **~1.1ms** | 单窗 (16 帧 × 51 维) |
| 测试总数 | **86** | 全部通过 |
| URFD fall-01 检出 | **track 3, 3.70-3.80s** | 多目标规则 smoke |
| URFD adl-01 误报 | **0** | 无事件输出 |

---

## 三、已解决的 P0/P1 问题

### P0-1: 评测协议自定义假设 → 已扩展双模式

- **问题**: 只有 event 模式 (temporal IoU=0.3)，若官方采用 clip 级二分类则不一致
- **解决**: `eval/metrics.py` 新增 `clip_pr_curve` 和 `competition_map(mode="clip")`
- **验证**: `tests/test_metrics.py` 新增 2 个 clip-level 测试, 1 个 `map_percent` 测试

### P0-2: fall/fallen 语义错误 → 已修正

- **问题**: 原来把 `fallen` 当作 `hard_negative`，会降低倒地后报警概率
- **解决**: `tools/prepare_omnifall_events.py` 新增 `event_semantics()` 函数
  - `fall` → `fall_process`
  - `fallen` → `post_fall_state` (同属 `fall_incident`，但分开保存)
  - 两者都标记 `is_fall_incident=1`
- **验证**: `tests/test_data_manifest.py` 新增 `test_event_conversion_preserves_post_fall_as_incident_not_negative`

### P0-3: 数据泄漏风险 → 已建立无泄漏 manifest

- **问题**: 原转换脚本丢弃 subject/cam 信息，无法按被试划分
- **解决**: `tools/build_manifest.py` 新增 `assign_group_splits()` 函数
  - 按 `group_id` (subject/trial) 划分，同一试次的多机位只进同一集合
  - 确定性划分 (seed=2026)，按标签分层 (fall/non-fall)
- **验证**: `tests/test_data_manifest.py` 新增 4 个测试 (确定性、分层、无泄漏)

### P1-1: 端到端基线缺失 → 已实现

- **问题**: 只有评测脚本和事件聚合器，没有端到端推理入口
- **解决**: `infer.py` 实现完整流程
  - 输入: 视频路径 (支持 RGB / 灰度 / URFD 横向拼接)
  - 输出: 事件 JSON + 带骨架/分数/告警横幅的 MP4
  - 自动裁剪: 识别 URFD 的横向拼接帧
- **验证**: 实跑 URFD fall-01 (检出事件 3.63-3.80s) 和 adl-01 (无误报)

### P1-2: 物理规则误报 → 已抑制

- **问题**: 单纯"质心下坠"会把"走向摄像机"误判为跌倒
- **解决**: `pipeline/fusion.py` 采用"累计下坠 × 姿态变化"策略
  - 累计下坠: 1.2 秒窗口内的净向下位移
  - 姿态变化: 躯干由竖直转为倾斜 (verticality < 0.8)
  - 两者相乘: 只有"快速下坠 + 倾斜"共同出现才给高分
- **验证**: `tests/test_fusion.py` 新增 `test_temporal_scorer_rejects_walking_toward_camera_while_upright`

### P1-3: 主目标跟踪不连续 → 已实现

- **问题**: 无跟踪器，每帧独立检测会导致目标切换
- **解决**: `pipeline/pose_track.py` 实现 `PrimaryPoseTracker`
  - IoU 优先: 与前一框 IoU 最高的检测优先
  - 置信度兜底: IoU 都低时选置信度最高
  - `track_switched` 标记: 检测到目标切换时重置时序历史
- **验证**: `tests/test_pose_track.py` 新增 3 个测试 (连续性、切换标记、重置)

### P1-4: 时延测试波动 → 已改为 smoke test

- **问题**: TCN 时延测试偶发超过 5ms 阈值，阻断单元测试
- **解决**: `tests/test_tcn.py` 将 `test_latency_budget` 改为 `test_latency_smoke`
  - 只验证推理可执行且时延合理 (0 < dt < 1000ms)
  - 严格阈值测试移到 `eval/benchmark.py` (warm-up 后 P50/P95)
- **验证**: 86 测试全部通过

---

## 四、待办事项 (按优先级)

### 4.1 立即执行：冻结 Round 0 checkpoint

1. **提交当前改动**:
   ```bash
   cd /d/HermesWorkspace/Detictive
   git add -A
   git commit -m "feat: establish hardened multi-track fall baseline"
   ```

2. **提交前质量门**：全量测试、`git diff --check`、静态安全扫描、focused review。

3. **说明**：本 checkpoint 只证明多目标规则 smoke 可运行，不包含完整验证集准确率。

### 4.2 Round 1：三 worktree，小步并行

1. **批量评测优先**：模型只加载一次、`render=False`、完整 URFD val/test clip-level 指标。
2. **统一视频读取与 pose cache**：普通 MP4 + OF-Syn tar URI；先 4 个样例，再 20 个 smoke，暂不全量抽取。
3. **红外/低光增强纯函数**：独立 worktree，不接入训练主流程。

详细分支、文件所有权、验收条件和 Agent 提示词见：
`.hermes/plans/2026-08-17_015002-competition-gap-and-execution-plan.md`。

只有 Round 1 的缓存 schema 和完整规则 val 指标通过后，才启动 TCN Dataset 与训练。

### 4.3 D4-D9 (第 3-7 天)

1. **通道融合策略** (`pipeline/fusion.py` 升级):
   - 当前: 纯规则 (累计下坠 × 姿态变化)
   - 升级: 规则分数 × TCN 分数 (加权融合)
   - 扫参确定最佳权重

2. **误报分析**:
   - 导出 FP 片段，按动作类型/光照/视角聚类
   - 针对性补数据或调规则

3. **消融实验**:
   - 规则 only / TCN only / 规则+TCN
   - 各增强开关
   - 输出 `reports/ablation.md`

### 4.4 D10-D12 (第 8-10 天)

1. **轻量化冲刺**:
   - MGD 特征蒸馏 (教师 YOLO11m-pose → 学生 n-pose)
   - 输入分辨率扫描 (640 / 512 / 416)
   - 参数量压缩到 8-12M

2. **红外专项验证**:
   - 灰度域测试集上的指标单独报告
   - `reports/ir_eval.md`

3. **冻结算法版本 v1.0**:
   - 完整回归测试
   - 一键复现脚本

### 4.5 D13-D15 (第 11-13 天)

1. **提交材料**:
   - 项目文档 (按附件 2 模板)
   - 参赛作品简介 (≤300 字)
   - 项目视频 (≤5min, ≤200M, mp4)
   - 代码整理: README, 模型使用说明, 权重, requirements

2. **材料审查**:
   - 无校名/导师信息
   - 命名规范: `团队名_项目名_材料类型.扩展名`

---

## 五、已知风险与注意事项

### 5.1 评测协议未定

- **风险**: 官方尚未明确事件匹配协议 (event vs clip, temporal IoU 阈值)
- **缓解**: `eval/metrics.py` 已支持双模式，待官方确认后切换
- **行动**: 尽快向赛题专家确认 (邮箱: jianzhong.he@huawei.com)

### 5.2 NPU 20MB 口径不明

- **风险**: 赛题要求"NPU 存储占用 ≤20MB"，但补充说明改为 V100 测试
- **缓解**: 当前 fp32 权重 11.50MB，远低于约束
- **行动**: 确认 20MB 是指 fp32 权重还是运行时内存

### 5.3 OF-Syn 视频格式

- **风险**: OF-Syn 视频为 AV1 编码，部分播放器/库可能不支持
- **缓解**: OpenCV 5.0 支持 AV1; 实测 URFD 和 OF-Syn 样例均可正常解码
- **行动**: 如遇解码问题，用 ffmpeg 预转码

### 5.4 本机 GPU 显存限制

- **风险**: RTX 4060 Laptop 8GB，batch size 需控制在 16-32
- **缓解**: YOLO11n-pose + TCN 总参数 ~3M，显存占用低
- **行动**: 训练时监控显存，必要时降低 imgsz 或 batch size

### 5.5 OF-Syn 随机划分与多目标跟踪限制

- **OF-Syn split**：manifest 未发现 clip/group/path 跨 split，但上游缺少可用 subject/scene ID；只能称为随机 clip split，不能证明身份独立。
- **多目标跟踪**：当前为轻量 IoU + 常速度中心预测，无外观 ReID；首次交叉、复杂遮挡或长时间离场仍可能产生 ID fragmentation。事件保留 track_id 便于后续错误分析。
- **行动**：Round 1 只在 val 统计 track 数、切换、漏检和多人失败；test 保留到最终冻结后一次运行。

---

## 六、下一步行动指令

### 立即执行

1. 实现普通 MP4 与 OF-Syn tar 成员的统一视频源；
2. 导出关键点、bbox、track ID、timestamp、valid mask 的 `.npz` pose cache；
3. 先做 10-clip cache smoke 和重跑一致性，再扩展 OF-Syn val；
4. test split 继续封存，不用于阈值或融合权重选择。

---

## 七、关键文件索引

| 文件 | 用途 | 状态 |
|---|---|---|
| `eval/metrics.py` | 评测指标 (clip/event 双模式) | ✅ 已实现 |
| `eval/benchmark.py` | 模型参数与时延基准 | ✅ 已实现 |
| `eval/evaluate_manifest.py` | 显式 split 批量评测与断点缓存 | ✅ 已实现 |
| `infer.py` | 端到端视频推理 | ✅ 已实现 |
| `pipeline/inference_engine.py` | 单模型复用、无渲染推理、分阶段时延 | ✅ 已实现 |
| `models/tcn.py` | 0.134M 因果 FallTCN | ✅ 已实现 |
| `pipeline/pose_track.py` | 多目标姿态轨迹生命周期 | ✅ 已实现 |
| `pipeline/rules.py` | 姿态物理特征与规则分数 | ✅ 已实现 |
| `pipeline/fusion.py` | 时序规则融合 | ✅ 已实现 |
| `pipeline/event_aggregator.py` | 事件聚合器 | ✅ 已实现 |
| `tools/build_manifest.py` | 无泄漏视频 manifest | ✅ 已实现 |
| `tools/prepare_omnifall_events.py` | OmniFall 标注转换 | ✅ 已实现 |
| `tools/augment.py` | 数据增强套件 | ❌ 待实现 |
| `tools/extract_keypoints.py` | TCN 训练数据提取 | ❌ 待实现 |
| `scripts/train_tcn.py` | TCN 训练脚本 | ❌ 待实现 |
| `configs/` | 训练/推理配置 | ❌ 待实现 |

---

## 八、环境信息

- **OS**: Windows 10
- **Python**: 3.11.14
- **PyTorch**: 2.13.0+cu126
- **Ultralytics**: 8.4.120
- **GPU**: RTX 4060 Laptop 8GB (CUDA 13.2)
- **磁盘**: D: 753G, 已用 668G, 剩余 85G

---

## 九、交接检查清单

- [x] 86 测试全部通过
- [x] URFD val 14/14，P@R90/P@R95=0.80/0.80，test 未使用
- [x] URFD fall-01 检出 track 3 事件 (3.70-3.80s)
- [x] URFD adl-01 无误报事件
- [x] YOLO11n-pose 2.874M 参数 (远低于 20M 约束)
- [x] fp32 权重 11.50MB (远低于 80MB 约束)
- [x] manifest 未发现直接 group/clip/path 跨 split；OF-Syn 身份独立性仍未证明
- [x] fall/fallen 语义修正 (都属于 fall_incident)
- [x] 评测器支持 clip/event 双模式
- [x] OF-Syn 12,000 个 tar URI 全部命中实际成员；manifest 无非法事件区间
- [x] 多目标轨迹独立评分，事件保留 track_id；缺失检测使用 unknown 语义
- [x] README.md 和 requirements.txt 已创建
- [x] 一键复现脚本 `scripts/reproduce.sh` 已创建
- [ ] 当前改动已提交 (待执行)
- [ ] 数据增强套件已实现 (D2)
- [ ] TCN 训练数据已准备 (D2)
- [ ] TCN 已训练 (D3)
- [ ] 消融实验已完成 (D4-D9)
- [ ] 提交材料已准备 (D13-D15)

---

**文档生成者**: Hermes Agent  
**最后更新**: 2026-08-16 Round 0 复审后  
**下次审查**: Round 0 checkpoint 提交后、创建 worktree 前
