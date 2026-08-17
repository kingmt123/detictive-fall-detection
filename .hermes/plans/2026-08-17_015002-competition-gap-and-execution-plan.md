---
title: Detictive 赛题合规差距与两周执行计划
created: 2026-08-17 01:50
status: active
source_of_truth: 面向低算力端侧平台基于视觉的实时跌倒检测.docx
supersedes_execution_order: 2026-08-16_234916-next-stage-multi-agent-plan.md
---

# Detictive 赛题合规差距与两周执行计划

## 0. 执行结论

当前项目已完成一个可运行、可测试的 **YOLO11n-pose + 轻量多目标跟踪 + 姿态运动规则 + 事件聚合** 基线，但还不是可提交的竞赛候选系统。

下一步最高优先级不是继续堆模型，而是建立 **完整验证集批量推理与可复现指标闭环**。原因：竞赛准确率占 50%，当前没有任何完整 val P@R90、P@R95、MAP、FP/FN；在没有误差证据前增加第二 YOLO、蒸馏、剪枝或量化都无法判断收益。

执行顺序冻结为：

1. Round 0 checkpoint；
2. 可复用推理引擎 + URFD val 批量评测；
3. 普通 MP4/OF-Syn tar 统一视频源 + pose cache；
4. TCN 训练与 val 调参；
5. 规则/TCN/融合消融与一次性 test；
6. 1080P 端到端时延、V100 复测与提交材料。

## 1. 权威要求与当前证据矩阵

| 赛题要求 | 当前证据 | 状态 | 阶段门 |
|---|---|---:|---|
| 端到端纯视觉，不依赖深度/穿戴设备 | `infer.py` 只读取 RGB/RGB半幅；YOLO11n-pose + 规则 | ✅ 基线满足 | 批量评测保持纯视觉输入 |
| 支持红外图像 | 只有 URFD 灰度/彩色 smoke；无红外专项数据、增强、指标 | ❌ | Round 1C 建立灰度/低光/噪声/遮挡增强和 IR-like val slice |
| 相机输入 1080P 以上，算法输入可自定 | `test_auto_crop_accepts_1080p_camera_input_without_splitting` 验证 1920×1080 预处理；模型内部 640 | 🟡 接口级满足 | Round 4 跑真实/合成 1080P 视频端到端 P50/P95 |
| 模型参数 ≤20M，FP32 参数量 ≤80MB | 实测 YOLO 2,874,462 + TCN 133,697 = 3,008,159 参数；FP32 参数约 12.03MB | ✅ 结构满足 | 训练后对最终 checkpoint 重新统计，报告口径固定 |
| 推理耗时 ≤100ms | RTX4060 `model.predict()` 微基准 P50 47.92ms/P95 52.82ms；完整链路未知 | 🟡 高风险未闭环 | Round 1 先加分阶段计时，Round 4 在无渲染与1080P输入上测端到端；最终 V100 复测 |
| 推理时 NPU 存储占用 ≤20MB | 测试硬件已改为 V100，但 DOCX 称核心指标不变；当前无 NPU runtime/activation 口径 | ⚪ 未验证 | 向主办方确认“存储”口径；材料只报告参数/权重，不冒充 NPU 峰值 |
| 每片段可含一个或多个跌倒 | `aggregate_tracks` 可输出多事件/多 track；仅单 fall smoke | 🟡 代码支持、数据证据不足 | 批量评测增加多事件 fixture；OF-Syn event val 验证多事件 |
| P@R90、P@R95、MAP | `competition_map(mode=...)` 有 18 个严格测试；无完整数据结果 | 🟡 评测器就绪 | Round 1 产出 URFD val clip-level 指标；event IoU 仅作本地代理并显式标注 |
| 公开+非公开测试代码 | 只有单视频 CLI；无批量、缓存、恢复、稳定输出契约 | ❌ | Round 1 必须产出 `eval/evaluate_manifest.py` 和使用说明 |
| 纯端侧/隐私闭环方案说明 | 当前架构可本地运行，但尚无正式项目文档 | 🟡 | 提交文档说明所有视频在本地/V100测试进程内处理 |
| 创新性、合理性、业界对比（专家30%） | 有调研文档与轻量姿态时序路线；无完整消融和对比表 | 🟡 | Round 3 产出规则/TCN/融合消融，Round 5 形成可引用表格 |
| 300字内简介 PDF | 无 `submission/` / `deliverables/` | ❌ | 2026-08-29 前完成匿名版 |
| 项目文档 PDF | 无正式模板文档/PDF | ❌ | 2026-08-29 前完成，包含数据、算法、硬件、引用来源 |
| ≤5分钟、≤200MB MP4 | 有若干 smoke MP4，但无成片、旁白、指标页 | ❌ | 2026-08-30 前完成 3–4 分钟演示 |
| 其他 ZIP ≤200MB | 无可移植提交包、批量入口、模型说明 | ❌ | 2026-08-30 前完成并在干净目录验收 |
| 材料匿名，不出现学校/学院/导师 | 当前核心文档未出现团队身份；原始论文抓取含第三方作者单位，不应进提交包 | 🟡 | 包装时白名单复制，不直接压缩整个仓库；执行匿名扫描 |

## 2. 当前成熟度评估

### 2.1 已经可信的部分

- 74 个单元测试覆盖指标、数据语义、因果聚合、unknown、时间归一、多目标跟踪、TCN 因果性和 1080P 预处理。
- `data/manifest.csv` 实际生成 12,100 行：OF-Syn 12,000、URFD 100。
- OF-Syn 12,000 个 `tar://...!/./...` 成员均可在 9.7GB 归档中命中；非法事件数为 0。
- fall-01 smoke 输出 track 3、3.70–3.80s；adl-01 输出 0 事件。
- 轻量性有实测参数证据，距离 20M/80MB 限制有充分余量。

### 2.2 不能宣称完成的部分

- 73 tests 不是竞赛准确率。
- fall-01/adl-01 不是完整公开验证集。
- 47.92ms 是 `model.predict()` 微基准，不是单帧端到端耗时。
- TCN 只有随机初始化网络结构，没有 cache、训练 checkpoint 或 val 指标。
- OF-Syn 随机 split 只能证明 manifest 字段层不交叉，不能证明 subject/scene/template 独立。
- event IoU=0.3、一对一匹配和 P@R 插值是本地代理协议，不能写成主办方已确认规则。
- NPU 存储峰值没有实测。
- 轻量跟踪器不是 ReID；首次交叉、复杂遮挡和长离场仍可能换 ID。

## 3. 评分导向的优先级

| 优先级 | 工作 | 为什么现在做 | 停止条件 |
|---:|---|---|---|
| 1 | 批量评测 + FP/FN | 直接闭环 50% 准确率主分，决定后续所有模型工作 | URFD val 所有 clip 均有预测，指标/错误清单可复现 |
| 2 | pose cache + TCN | 时序语义是区分跌倒与坐/躺的核心，参数增量仅 0.134M | val 指标相对规则无收益则停止扩大 TCN |
| 3 | 红外/低光/遮挡增强 | DOCX 明确要求支持红外与长尾场景 | val slice 无收益或显著伤害正常域时停止 |
| 4 | 完整端到端时延 | 25% 评分且必须 ≤100ms | P95 超标才优化；达标则不提前量化 |
| 5 | 跟踪器升级 | 仅在多人/遮挡错误中有证据才有收益 | ID switch 不是主要 FN/FP 来源时不做 ReID |
| 6 | 第二外观 YOLO | 增加参数/时延，且当前无误差证据 | 只有姿态不可观测造成主要 FN 且时延仍有余量才试 |
| 7 | 蒸馏/剪枝/量化 | 当前 3.008M 已很小，先做可能浪费时间并损害精度 | 只有 V100端到端超100ms或明确有额外评分收益才做 |

## 4. 依赖感知执行轮次

### Gate 0 — 2026-08-17：冻结 Round 0

验收：

- `python -m pytest -q` 全绿；
- `ruff check .` 全绿；
- `git diff --check` 无错误；
- fall/ADL smoke 维持一正一负；
- focused review 无 P0/P1；
- 提交 `feat: freeze deterministic evaluation and multitrack rule baseline`。

### Gate 1 — 2026-08-17 至 2026-08-20：先得到完整 URFD val 基线

分支/worktree：`feat/batch-eval`

状态：**2026-08-17 已完成实现、真实 val 验收与 focused review；无剩余 P0/P1。**
证据：`reports/urfd_val_r0_metrics.json`；URFD val 14/14、P@R90/P@R95=0.80/0.80、
本地 clip MAP=80.0%，86 tests；test split 默认拒绝，并有固定持久化 seal 防止成功后重跑。

交付：

- `pipeline/inference_engine.py`：模型只加载一次；`analyze(video, render=False)`；输出稳定 schema；
- `eval/evaluate_manifest.py`：只允许显式 split/mode；JSONL/CSV 结果缓存；断点续跑；失败 clip 不得静默补零；
- 分阶段时延：decode/crop、predict、CPU transfer、track/rule、aggregate；渲染和编码单列；
- URFD val clip-level P@R90/P@R95/MAP、混淆、FP/FN clip 清单；
- 禁止读取 test 结果调阈值。

验收：

- 单元测试用 fake engine 证明模型单次加载、resume、失败即非零退出、每 GT clip 均有预测；
- 真实 URFD val 14 个 clip 全部完成；
- 报告记录阈值来源、硬件、环境和 commit；
- `render=False` 结果与单视频 wrapper 的 clip score/event 一致。

### Gate 2 — 2026-08-19 至 2026-08-23：统一视频源与 pose cache

分支/worktree：`feat/pose-cache`，可与 Gate 1 并行写代码，但 GPU 任务排队运行。

交付：

- `pipeline/video_source.py`：普通 MP4 与 `tar://archive!/member` 统一读取；临时文件生命周期安全；
- `tools/extract_keypoints.py`：可恢复、原子写入 `.npz`；保存时间戳、17×3 keypoints、bbox、track_id、valid mask、fps、源指纹；
- cache schema/version 和 manifest 对齐审计；
- 先 2 个 URFD + 2 个 OF-Syn 小样本 smoke，再扩 val；不立即跑完整 12,000。

验收：

- tar/普通视频测试；损坏 cache 不会被误当成功；
- 相同输入和 seed 生成确定性元数据；
- 训练代码不允许每个 epoch 重跑 YOLO。

### Gate 3 — 2026-08-23 至 2026-08-26：TCN 与长尾增强

交付：

- pose-window Dataset、padding/mask、标签对齐；
- `train_tcn.py`、配置、seed、checkpoint、val 曲线；
- 规则 only、TCN only、规则+TCN 三组；
- 灰度/低光/噪声/遮挡增强仅作用于 train，生成可视化审计网格；
- 只在 val 选择阈值/融合权重。

阶段门：

- 若 TCN 在 val 的目标 MAP 或 P@R95 无可重复提升，保留规则基线并停止扩模型；
- 若红外增强损害正常域且 IR-like slice 无提升，回退增强强度；
- 不在此阶段加入第二外观 YOLO。

### Gate 4 — 2026-08-26 至 2026-08-28：冻结候选与一次性 test

交付：

- 冻结所有阈值、权重、窗口和后处理；
- 在 test 上只运行一次；
- 输出 clip/event（若GT可用）指标、FP/FN、多人/遮挡/低光切片；
- 比较总参数、FP32体积、权重文件体积和完整端到端时延。

只有以下证据之一成立才开启外观通道实验：

1. 主要 FN 来自姿态完全不可观测；
2. TCN/规则对这些样本无法恢复；
3. V100 P95 与参数预算仍有明确余量；
4. 有独立 val 证明收益，不用 test 选方案。

### Gate 5 — 2026-08-28 至 2026-08-31：性能与提交材料

- 1080P 视频、`render=False` 端到端 P50/P95；V100 环境复测；
- 明确区分单帧计算时延、在线确认延迟和离线编码耗时；
- 300字内简介 PDF；
- 项目文档 PDF：设计、数据、来源、引用、指标、消融、局限、部署；
- 3–4 分钟 MP4，≤200MB；
- 白名单 ZIP，≤200MB，含代码、权重、模型说明、批量入口；
- 匿名扫描：团队材料不得出现学校、学院、导师或个人路径；
- 清除 DOCX/PDF 作者、最后修改者等元数据；原始赛题 DOCX 自带作者字段，不能直接作为匿名项目模板另存；
- 2026-08-31 完成首次提交，保留 24 小时缓冲，最终截止 2026-09-01 23:59。

## 5. 并行编排

- 主线/Agent A：批量推理引擎、URFD val 报告。
- Agent B：视频源抽象、tar 读取、pose cache schema。
- Agent C：红外/低光/遮挡增强，只写代码和小样本审计；不得与 A/B 同时跑大 GPU 批任务。
- Reviewer：每个 worktree 合并前只读 focused review，先 P0/P1，再全量测试。
- GPU 队列顺序：URFD val 基线 → pose cache smoke → pose cache val → TCN train → 最终 benchmark。

## 6. 立即执行的下一小步

checkpoint 后只启动 `feat/batch-eval`：

1. 先写 fake-model 测试，要求模型只构造一次；
2. 从 `infer.py` 提取可复用 engine，并支持 `render=False`；
3. 保留原单视频 CLI 作为兼容 wrapper；
4. 在 fall-01/adl-01 做等价回归；
5. 再实现 manifest 批处理与 resume；
6. 只跑 URFD val，不跑 test。

这一步完成前，不启动完整 OF-Syn cache、TCN 训练、第二 YOLO、蒸馏、剪枝或量化。

## 7. 风险登记

| 风险 | 影响 | 缓解 |
|---|---|---|
| 官方 TP/FP/FN 与 P@R 插值未明确 | 本地 MAP 可能与官方不一致 | 保持 clip/event 显式模式；联系主办方；报告本地假设 |
| URFD 无 event GT | 无法验证 temporal IoU | URFD 只做 clip；OF-Syn 做本地 event proxy |
| OF-Syn 随机 split 潜在模板泄漏 | val 过乐观 | 文档披露；尽可能提取场景/来源字段做分组分析 |
| 跟踪 ID switch | 多人 FP/FN | 统计后再升级；不预先引入重型 ReID |
| PyTorch 依赖/权重下载导致隐藏环境失败 | 非公开测试不可运行 | 锁定环境、离线权重、启动自检、清晰失败信息 |
| 当前无 Git remote | checkpoint 无异地备份 | checkpoint 后配置私有远程或离线压缩备份 |
| 提交材料包含原始研究抓取中的第三方单位信息 | 匿名审查歧义/包过大 | 最终 ZIP 使用白名单，不包含 `docs/research/_raw` |

## 8. 完成定义

项目只有同时满足以下条件才可称“初赛候选”：

- 完整 val 及一次性 test 的 P@R90/P@R95/MAP 有可复现报告；
- FP/FN 和长尾切片有分析；
- 最终模型参数、FP32体积、权重体积有实测；
- 1080P 输入下完整 `render=False` 端到端 P95 有实测，并在目标环境 ≤100ms；
- 纯视觉、红外支持和多事件行为有证据；
- 批量入口能在干净环境运行，失败不静默；
- 四项提交材料齐全、匿名、格式和大小合规。
