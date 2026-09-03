# 项目迭代日志

## 2026-08-24 — 70% 模型瓶颈调研

### 完成事项

- 审计 `MultiStreamMultiScaleAttentionTCN`、MIL 训练器、三 seed 配置、G2 决策卡和历史 controlled-improvement 结果。
- 确认当前依赖闭包已恢复，模型和训练模块可正常导入。
- 识别训练 top-k 聚合与验证 max 聚合不一致，以及 BCE 与 P@R90/P@R95 指标错位。
- 检索并核验弱监督时序定位、AUPRC 优化、细粒度骨架对比学习、PoseConv3D、现实姿态错误增强和跨域泛化文献。
- 形成激进改进文档：`docs/research/2026-08-24_aggressive_70pct_breakthrough_plan.md`。

### 遇到的问题

- 历史约 69.63% checkpoint 缺少原始 manifest/split hash，不能与当前冻结 validation 严格横比。
- hard-negative ranking 只带来小幅 mAP 改善，同时损伤 P@R90 并增加 `lie_down` FP。
- 已预注册 G2 尚未运行，后续激进方案不能改写或污染其控制协议。

### 解决方案

- 将首选突破路线定为统一聚合器、指标对齐损失和受控过渡专用 verifier。
- 将 PoseConv3D/RGB 教师蒸馏作为第二条高风险高收益路线，保持部署学生轻量。
- 为每条路线定义单 seed canary、三 seed 均值、worst-seed、活动级 FP 和停损条件。

## 2026-08-24 — HFlip 三种子完成后的 80% 突破复盘

### 完成事项

- 核验固定四模型 logit 集成：mAP 71.1794%、P@R90 74.0741%、P@R95 68.2848%。
- 由高召回指标反推误报目标：R90 至少减少 20 个 FP，R95 至少减少 46 个 FP 才能达到 80% precision。
- 对照数据与训练代码，确认项目已有逐帧事件区间和阶段语义，但主训练仍采用视频级 top-k MIL、验证采用 max-window 聚合。
- 调研 MS-TCN、ASRF、ActionFormer、PoseConv3D、class-aware contrastive learning、AUPRC 优化与跨模态时序检测蒸馏。
- 形成新报告：`docs/research/2026-08-24_post_hflip_80pct_breakthrough_report.md`。

### 遇到的问题

- 同质 HFlip 多 seed 集成仅提升 0.9019 个百分点，已进入收益递减区。
- 简单事件 GRU、冻结 R3D-18 教师和轻量姿态扰动均未展示稳定增益。
- 项目 validation 已用于多轮方案筛选，不能继续作为完全未触碰的最终选择集。

### 解决方案

- 将主线升级为利用现有强时序标签的因果事件检测：阶段分割、边界回归、事件质量评分和合法事件聚合。
- 先建立长上下文骨架、PoseConv3D、事件对齐 RGB 三层 oracle，判断 80% 的信息上限，再决定蒸馏或补充模态。
- 所有研发改用 train-only 分组 OOF，并为事件基线、hard-negative verifier、多尺度上下文和蒸馏定义量化通过/停损线。

## 2026-08-25 — 当前最优架构与 90% 指标重新定义

### 完成事项

- 还原当前最优完整链路：YOLO11n-Pose、轻量多人跟踪、六流多尺度因果 TCN、阶段辅助头和四模型固定 logit 集成。
- 实例化当前配置确认单模型参数量为 161,406，四模型时序部分合计约 645,624 参数。
- 发现当前 71.1794% 为 mAP，不能与外部论文的 Accuracy 横比。
- 由 1,200 条 validation 和 P@R 指标反推：当前系统在 R90/R95 工作点的普通 Accuracy 已约为 92.33%/90.92%。
- 核验 TCNTE、FallNet、OmniFall、LTC-Fall、真实姿态错误增强和 SkeleTR 原始文献。
- 形成报告：`docs/research/2026-08-25_current_best_architecture_and_90pct_plan.md`。

### 遇到的问题

- 上一版事件检测与 oracle 主线已被验证不可行，不能继续作为改进前提。
- 新调研混用了 Accuracy、mAP、Precision 和主观工程置信度，若直接比较会错误判断项目差距。
- OmniFall 表明 staged 数据集高 Accuracy 不能外推到 staged-to-wild 真实事故域。

### 解决方案

- 普通 Accuracy 90% 改为阈值锁定和独立验证问题，不为此重写模型。
- 将有效 90% 合同定义为 Recall ≥95%、F1 ≥90%、Precision ≥85%，mAP 90%仅作为 stretch。
- 新主线最大程度复用当前模型：16+48 双尺度因果 TCN、融合后单层 Transformer、三状态主头、真实 pose-error 增强和 track quality token。
- 为每项改动设置单变量 OOF 对照、三 seed 通过线和停损条件。

## 2026-08-25 — clip mAP 90% 架构专项调研

### 完成事项

- 将目标明确为 clip mAP ≥90%，确认相对当前 71.1794% 仍有 18.8206 个百分点缺口。
- 核验 PG-ProtoFall、TubeLite、ContextDet、class-aware contrastive、SOAP/AUPRC 和 OmniFall 等原始文献。
- 确认近期最直接的新证据来自 RGB appearance + pose 显式融合、监督对比和 prototype regularization。
- 设计 EdgeFall-X90：复用 YOLO11n-Pose 特征图的 person/context ROI 分支、六流双尺度 TCN、pose-quality gated cross-attention 和多任务原型头。
- 形成报告：`docs/research/2026-08-25_map90_architecture_upgrade_plan.md`。

### 遇到的问题

- mAP90要求整条 PR 排序曲线改善，不能由普通 Accuracy 90% 或单个阈值结果替代。
- 当前纯骨架模型在 R90/R95 下仍约有 70/98 个 FP，需要新增外观与场景信息才能大幅削减。
- 公开时序动作检测和 OmniFall staged-to-wild 结果均不支持“增加模型容量即可保证 mAP90”。
- 上一版事件分割/oracle、冻结 R3D-18、简单 GRU 和同质多 seed 路线均不能重复作为主线。

### 解决方案

- 不新增第二个完整 RGB backbone，复用 YOLO11n-Pose P3/P4 特征进行 actor/context ROIAlign，控制端侧成本。
- 使用 pose-quality 连续门控融合 skeleton 与 appearance，针对关键点缺失和跟踪错误。
- 将 `lie_down/lying/stand_up/sit_down` 作为结构化 subtype，以 class-aware SupCon 和 prototype loss 塑造融合 embedding。
- 先做简单 RGB ROI late-fusion canary；达不到 +3 mAP 即停止复杂融合，转向数据和标签闭环。
- 设置 76/82/86/88/90 五级里程碑和明确停损线，所有研发只看 grouped train-only OOF。

## 2026-08-25 — F2/F3 无效后的激进主干换代

### 完成事项

- 将用户确认的 F2 Context ROI 与 F3 quality-gated fusion 无效记录为正式否决证据。
- 重新审视当前信息链，确认坐标关键点、手工六流、末时刻 embedding 和 max-window 构成连续信息压缩。
- 调研 PoseConv3D、VideoMamba、VideoMambaPro、VideoMAE V2、InternVideo2、MoViNets 与 MGMAE 原始论文。
- 设计 FallMamba-X：全帧 RGB + 关节/骨骼热图 token 早期融合、80/96帧 VideoMambaPro 和完整 clip pooling。
- 设计 InternVideo2/VideoMAE RGB 教师与 PoseHeatmap-VideoMambaPro 教师的异构 OOF 融合及端侧蒸馏路线。
- 形成报告：`docs/research/2026-08-25_map90_radical_backbone_replacement.md`。

### 遇到的问题

- F2/F3 说明在最终 embedding 旁补 ROI 特征或质量门控不能修复前端信息损失。
- 当前六流模型依赖 tracker、坐标确定值和末帧聚合，难以利用遮挡、接触和完整空间结构。
- 继续保持 16 万参数并要求 mAP90 缺乏现实证据。

### 解决方案

- 停止以坐标六流和 16+48 TCN 为新方案前提，改用 pose heatmap volume 保留空间结构与置信不确定性。
- 使用端到端 InternVideo2-S RGB canary 作为第一生死门；OOF mAP <76 即停止架构冲90承诺。
- 研究阶段允许异构强教师先达到目标，再蒸馏到 10–25M 的 FallMamba-X 或 MoViNet/causal SSM 学生。
- 去除 top-k MIL 和 max-window，完整 5–6 秒 clip 直接产生排序分数。

## 2026-08-25 — F1/F2/F3 无效后的互补专家融合重构

### 完成事项

- 将 F1（RGB–Pose late fusion）、F2（Context ROI）和 F3（quality-gated fusion）均记录为无效路线。
- 重新分析旧最佳有效原因：任务匹配的六流运动先验、小参数量、多尺度因果TCN、定向难负例采样、阶段辅助监督和多seed降方差。
- 将激进架构从“整体替代模型”重构为 R90/R95 局部 residual experts。
- 设计以冻结旧logit为offset、只在疑难分数带生效的受约束残差融合。
- 定义“单项工作点改善即可进入专家库”和“多指标保护后才能进入生产融合”的两级准入制度。
- 形成报告：`docs/research/2026-08-25_map90_complementary_expert_fusion_report.md`。

### 遇到的问题

- 以总体mAP筛选新架构会丢弃局部互补模型；一个总体较弱的RGB模型仍可能识别旧模型的特定高分FP。
- 全局late fusion要求新分支在所有样本上都可靠，F1/F2/F3已证明该假设不成立。
- 直接在当前validation搜索融合权重会进一步放大验证集过拟合。

### 解决方案

- 冻结旧四模型集成作为全局anchor，不重新训练，不丢失已验证的骨架运动能力。
- 以旧模型OOF分数构造R90/R95疑难带，分别训练RGB、pose-heatmap或subtype专家。
- 专家只需P@R90/P@R95提升2点，或在对应工作点减少预注册数量FP，即可进入可学习专家库。
- 最终融合采用nested grouped OOF、非负权重、残差幅度裁剪和带外严格回退旧分数。
- 第一优先实验改为R95 hard-negative RGB residual expert：目标减少≥15 FP且损失≤1 TP。

## 2026-08-25 — 沿原最佳模型的 Stage-S1-R2 改进报告

### 完成事项

- 重新核对当前三个配置和 `MultiStreamMultiScaleAttentionTCN` 实现，确认已落地最佳主体是 16 帧六流因果 TCN；16+48 帧仍是待验证升级，不是当前事实。
- 分析旧最佳有效来源：任务匹配的六流先验、多核膨胀因果 TCN、阶段辅助监督、定向困难负例采样和 HFlip 多种子降方差。
- 定位三个可直接修复的结构瓶颈：训练 top20% 与验证 max 聚合错位、末 token 信息压缩、六流仅在末端交互。
- 设计 Stage-S1-R2：旧 16 帧路径不动，以零初始化残差依次增加证据聚合、共享权重 16+48 上下文、跨流适配和四模型 OOF 蒸馏。
- 形成报告：`docs/research/2026-08-25_original_best_model_improvement_report.md`。

### 遇到的问题

- 历史方案稿将 16+48 帧写入目标架构，容易与当前 16 帧已训练模型混淆。
- 事件 GRU、全局 hard-negative ranking 和 F1/F2/F3 已无效，不能改名后重新作为主线。
- 同一骨架输入从 71.18% 直接达到 mAP90 缺乏证据，必须区分可验证阶段目标与远期 stretch。

### 解决方案

- 在报告中分离“已验证基线”和“建议改造”，所有新增模块均使用零初始化残差，训练起点严格回退旧输出。
- 第一优先修复训练/推理聚合不一致，再测试共享 encoder 的 16+48 帧上下文；不复制第二套六流主干。
- 使用 train-only grouped OOF 四模型预测做蒸馏，禁止同折 teacher 和 frozen validation 调参。
- 为每项改造设置三 seed、五折、绝对 TP/FP、paired bootstrap 和明确停损线，将 75–78% 定为主目标、80% 定为条件目标。

## 2026-08-28 — 本机 Stage-S1 训练前置准备

### 完成事项

- 使用本机已有 Python 3.11.15 建立项目隔离环境 `.venv`，按 `requirements.txt` 安装 `torch 2.13.0+cu126`、`torchvision 0.28.0+cu126`、`ultralytics 8.4.120` 等锁定依赖。
- 在 RTX 4060 Laptop GPU 上完成 CUDA 矩阵运算验证，确认 `torch.cuda.is_available() == True`。
- 核对 Stage-S1 初始化权重 SHA-256 为 `45a40f762d0f6c53b0b7aeb8f239b7260cdc5fb529776cce2b722afe131f8e0f`，与报告记录一致。
- 在 `runs/local_training_20260828/window_cache_v2` 新建隔离窗口缓存：train 356,303 windows，validation 129,929 windows；未覆盖旧失败缓存。
- 在 `runs/local_training_20260828/sidecars` 完成 train/validation 六维 bbox/observed sidecar，并通过 base-cache 元数据和窗口逐项核验。
- 成功构建 161,406 参数 Stage-S1，兼容加载初始化 checkpoint 的 474 个参数键；在 CUDA 上完成 `(8,16,57)` 输入的二分类与四阶段前向，输出有限，峰值显存约 33.01 MiB。
- 运行 Stage-S1、MIL 与 sidecar 关键测试：`30 passed in 3.81s`。

### 遇到的问题

- 报告对应的 Primary 与三个 HFlip 最佳 checkpoint 仍未在本机找到，因此当前只能准备复现或等待正确权重后微调，不能宣称已恢复最佳四模型。
- 本机训练缓存只覆盖 3,667 个 of-syn train clips、356,303 个窗口；报告原始 Stage-S1 使用 988,337 个原始训练窗口，当前数据覆盖不足以无差别复现 70.2776% 单模型结果。
- 旧 `runs/local_cache_20260824/e2-full-cache/training/window_cache` 曾因过时的 `WindowMemmapCache.samples` 属性访问失败，并存在不可读残留目录。

### 解决方案

- 新训练环境、缓存和 sidecar 全部使用独立路径，保留旧产物作为审计证据，不覆盖、不删除。
- 正式训练前先恢复缺失的完整 train pose cache 与哈希为 `c9da36cb...` 的 Primary checkpoint；若无法恢复，只能明确标记为从 `45a40f...` 初始化权重进行子集复现。
- 继续使用当前完整的 1,200-clip validation 仅做流程预检；在训练数据补齐前不启动正式 Stage-S1 续训，也不与历史 70.2776%/71.1794% 指标直接比较。
- 复查旧提取日志确认 9,600 个 of-syn train clips 曾全部成功生成；当前缺失 5,933 个文件属于后续本地缓存丢失，不是源数据或提取器失败。
- 使用 `lying/lying_ch_021` 完成单 clip 恢复 canary：81 帧、81 个观测、无失败；随后启动后台全量恢复，父进程 PID `47048`、实际 Python 子进程 PID `50720`，日志目录为 `runs/local_training_20260828/pose_recovery`。
- 全量恢复只补缺失 cache 并验证已有 cache；完成前禁止基于当前 356,303-window 子集启动正式训练，完成后必须重新生成 full-manifest audit、window cache 和 sidecar。
- 2026-08-28 22:23 后首轮后台恢复被外部中断，未产生 Python traceback，`stderr.log` 为空；停止时 train pose cache 为 5,236/9,600。
- 2026-08-29 04:37 从现有 cache 断点续作，父进程 PID `23052`、实际 Python 子进程 PID `29456`；使用独立的 `stdout_resume_20260829_0436.log` 与 `stderr_resume_20260829_0436.log` 保留中断审计链。
- 2026-08-29 用户要求暂停任务；已停止 pose cache 恢复进程，确认无残留 `extract_keypoints` 进程。已恢复的 5,236 个 train cache 完整保留，可从该数量继续断点恢复。

## 2026-09-01 — 最终七头模型部署

### 完成事项

- 将 `fall_detection_seven_head_runtime_candidate.zip` 隔离部署到 `deploy/final_model`，未覆盖工作区已有研究代码和未提交改动。
- 按包内 `MANIFEST.json` 校验全部文件的大小与 SHA-256；最终七头权重 SHA-256 为 `8dd8e96d2d6d9e7ba5b29ea41bf4a5b95f14222e094859531cbf55ba5289d47f`，YOLO 权重 SHA-256 为 `869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0`。
- 使用 `fall-01-cam1.mp4` 完成原始视频到七头融合 JSON 的端到端 CPU 烟测：160 帧全部处理成功，输出协议为 `edgefall_seven_head_label_blind_runtime_v1`，分数为 `0.17800459265708923`。
- 在根 `README.md` 增加最终模型运行命令；原规则基线入口保持不变。

### 遇到的问题

- 原 `.venv` 的 Python 启动器指向已被清理的 Hermes 临时运行时，无法启动。
- 本机现有 Python 3.13 为 CPU 版 PyTorch，且全局 `mpmath`、`sympy` 安装文件不完整；Ultralytics 默认用户配置目录也不允许当前沙箱访问。

### 解决方案

- 不覆盖失效的 `.venv`，改用当前可用解释器进行部署烟测；补装 `ultralytics==8.4.120` 并修复 `mpmath==1.3.0`、`sympy==1.14.0`。
- 将 Ultralytics 配置目录限定到项目 `reports/ultralytics_config`，避免依赖用户目录。
- CPU 端到端烟测总耗时约 `7967.45 ms`、约 `49.80 ms/帧`；正式 CUDA/V100 的 P95 仍需在指定环境复测，不能由本次 CPU 烟测替代。

## 2026-09-01 — 最终模型比赛文档与演示视频

### 完成事项

- 基于冻结的七头运行包生成匿名项目简介、项目文档及 PDF 版本，内容使用 `MANIFEST.json` 中的固定七头融合规则和已核验验证指标。
- 生成 1280×720、41.33 秒的匿名项目视频，包含架构说明、原始 URFD 视频在线推理片段、验证指标和交付说明。
- 项目简介和项目文档均通过可访问性检查（0 high / 0 medium / 0 low）；Word 导出的 PDF 已逐页渲染并进行视觉检查。

### 遇到的问题

- 仓库已有 Gate 5 工具对应历史 TCN-only 候选，不能用于最终七头模型的材料。
- 文档渲染器依赖 LibreOffice，但本机未安装。

### 解决方案

- 新增独立交付物生成脚本，避免复用旧模型的 test 或 V100 数值；材料只呈现七头运行包的验证与烟测证据。
- 使用 Microsoft Word 导出 PDF，并通过 Poppler 渲染为 PNG 进行最终版式验收。

## 2026-09-01 — 官方模版项目文档与全流程检测视频定稿

### 完成事项

- 依据《第八届中国研究生人工智能创新大赛项目文档模版》重构为 8 页 A4 项目文档，覆盖项目概况、项目规划、实施方案、变更历史和参考资料；封面已填写团队“抹茶吃不饱”和“企业赛题七”。
- 正文从第 1 页连续编号，目录、表格、页眉和页脚与官方模版结构对应；最终 Word 文档与 PDF 均输出至 `deliverables`。
- 演示视频升级为 1280×720、30 FPS、40.67 秒的检测证据链：原始 RGB、17 点姿态骨架、人体框、track_id、候选时间区间、七个成员 raw logits 和融合风险分数同步呈现。
- 最终项目文档已完成 PDF 逐页视觉检查与可访问性审计（high / medium / low 均为 0）；视频已通过编码、分辨率、帧数和时长检查（1220 帧）。

### 遇到的问题

- 初版新文档的页眉页脚继承自前一节，造成项目名称和页码重复。

### 解决方案

- 将各节页眉页脚显式解除继承并清空后重建；正文页码改为 `PAGE` 域并从 1 重启，重新导出 PDF 与渲染核验。

## 2026-09-01 — 比赛提交材料最终完善

### 完成事项

- 项目文档补充技术路线图、结果展示图、数据/行业知识/算法/硬件来源表、技术调研对比、参考数据集与参考文献；最终版为 9 页，PDF 已逐页渲染检查。
- 生成 295 字参赛作品简介 PDF，并保留团队“抹茶吃不饱”、参数组别“企业赛题七”和项目名称的一致命名。
- 输出 9 页项目汇报 PPT；幻灯片覆盖应用目标、技术路线、创新点、来源合规、推理指标、实际检测画面和工程价值，并通过版式溢出检查。
- 将 80 秒 PPT 展示与 40.67 秒全流程检测演示拼接为最终项目视频：H.264、1280×720、30 FPS、120.67 秒、约 3.13 MiB；抽帧核验了封面、技术路线、指标、原始视频、姿态骨架、七头 logits 与收尾页。
- 生成“其他”辅助材料 ZIP，包含技术可行性与尽调说明、运行复核说明、冻结清单、端到端烟测结果、运行说明及两张核心图；压缩包为 168,388 字节并完成解压核验。

### 遇到的问题

- 项目视频若只保留检测片段，覆盖范围不足以同时呈现开发流程、创新要点与检测证据。

### 解决方案

- 使用 PPT 作为前置讲解段，再衔接同一最终运行包产出的检测演示段，形成“方案—指标—在线检测”的完整证据链；所有提交文件均按赛事要求的团队名与项目名命名。

## 2026-09-01 — 学术化文档、汇报 PPT 与动态演示增强

### 完成事项

- 项目文档升级为 V1.1、共 11 页（正文 8 页），补充双尺度七头架构图、模型选择收敛图和端到端运行证据图；详细说明数据来源、对照路线、创新性、验证协议、工程复现与风险控制。
- 模型选择图严格采用现有五折 train-OOF 配对审计结果；因冻结运行包不含逐 epoch 损失日志，未虚构训练损失曲线，并在正文中明确证据边界。
- 项目文档通过逐页渲染检查及可访问性审计，3 张图片均已补充替代文本，high / medium / low 问题均为 0。
- 项目汇报 PPT 重制为 12 页学术型演示稿，采用结论式标题、统一网格、原生数据图表与证据页脚；最终版通过幻灯片元素越界检查。
- 项目视频重制为 104.67 秒：前 42 秒为方案与证据讲解，后续约 58.67 秒集中展示原始 RGB、姿态骨架、并排对照和回放，最后 4 秒收束结论。输出为 H.264、1280×720、30 FPS、约 8.28 MiB。

### 遇到的问题

- 旧视频的现场演示片段较短，且静态讲解画面占比高，难以充分展示跌倒动作、姿态跟踪和检测结果之间的动态对应关系。
- 文档缺少可直接复核的学术图表；现有冻结包也不具备完整逐 epoch 训练日志。

### 解决方案

- 将同一检测样例拆分为减速原始视频、减速姿态结果、同步并排对照和两次回放，使动态检测内容超过视频总时长的一半；通过相邻抽帧像素差和关键时间点抽帧确认画面连续运动。
- 用真实的六头/七头配对增益、逐折差异、候选结构搜索及运行烟测结果构成学术证据图，避免把缺失日志包装为训练收敛曲线。

## 2026-09-01 — 动态样例与 APA 参考资料修订

### 完成事项

- 将中段静态“现场演示”页面移出视频时间线；说明段缩短为 28 秒，随后立即进入连续动态检测画面。
- 新视频按 fall-10、fall-04、fall-02、fall-01 四个独立 URFD 样例组织，其中前两段展示人体框、17 点姿态和候选事件，后两段展示不同动作过程与最终七头运行包的原始视频证据。
- 项目视频最终为 H.264、1280×720、30 FPS、68.50 秒、6.53 MiB；抽帧确认 0:28 起即有可见人体动作与姿态跟踪。
- 项目文档的参考资料已精简为 2 个数据集来源和 2 篇相关论文；采用 APA 第 7 版的作者-年份文内标注、按作者字母排序与悬挂缩进参考文献表。
- 更新后的 DOCX/PDF 仍为 11 页，完成逐页渲染检查和可访问性审计（high / medium / low 均为 0）。

### 遇到的问题

- 原 fall-01 演示片段在动作进入画面前有数秒空场，即使视频正常播放，也容易被观看者判断为“画面不动”。

### 解决方案

- 将姿态跟踪稳定的 fall-10 样例前置，并保留 fall-04、fall-02 与截取到跌倒发生点的 fall-01 作为连续的多样化动态展示；不再重复播放同一段原始视频。

## 2026-09-01 — 保留原视频并追加多案例演示

### 完成事项

- 恢复原 104.67 秒视频的完整内容：42 秒 PPT 说明、58.67 秒原始检测证据链和 4 秒结尾页均未删减。
- 在原检测证据链与结尾页之间追加 fall-10、fall-04、fall-02 和 fall-01 的独立动态样例；最终视频时长 141.17 秒，H.264、1280×720、30 FPS、约 14.39 MiB。
- 已抽帧核验原有静态说明页、原有检测动作、四段新增姿态/动作样例及结尾页。

### 遇到的问题

- 前一版为了让动态内容更早出现，缩短了原有 PPT 段和原有检测回放，违背了“只增加、不减少”的交付约束。

### 解决方案

- 以原视频时间线为基准完整复原，并仅在原检测段结束后追加新案例，再接回原结尾页。

## 2026-09-01 — 模型架构与结论章节细化

### 完成事项

- 将 3.1 节扩展为端到端架构分层说明，明确 RGB 输入、17 点姿态、轨迹关联、六类骨架特征、共享 ROI、16/48 帧时序分支及七头输出之间的数据流。
- 在 3.3 节补充成员头配置、ROI 尺度、因果窗口、logit 平均与 sigmoid 融合公式，使融合规则可复算。
- 扩展第 4 节，补充模型贡献、风险发现—人工复核—事件留档流程、职责分工表、隐私边界与后续独立验证计划。
- 更新目录页码，参考资料独立成页；最终导出 13 页 PDF，并完成逐页视觉检查和无障碍审计（高/中/低问题均为 0）。

### 遇到的问题

- 架构扩写曾导致 3.4 节末尾单独分页和参考资料断裂，影响文档紧凑性。

### 解决方案

- 压缩与结构表重复的文字、移除多余分页，并将参考资料显式置于新页；重新导出后确认表格、图注、目录与参考文献均未截断。

## 2026-09-03 — 比赛最终版本 GitHub 发布

### 完成事项

- 按项目所有者确认，将现有 `deploy/final_model` 定为比赛项目最终版本；README 增加最终部署命令、交付入口及验证边界，早期说明标为历史记录。
- 纳入冻结部署源码、锁定依赖、两份模型权重、MANIFEST、最终参赛文档/PPT/视频及烟测记录。
- 校验清单内 174 个文件的大小与 SHA-256 全部一致；170 个 Python 文件语法检查通过，部署文本未发现常见凭据模式。

### 遇到的问题

- 原有 `*.pt` 忽略规则会漏掉部署权重，Windows 自动换行转换可能破坏冻结文件哈希；工作区同时包含实验工作树和其他未提交训练改动。

### 解决方案

- 仅对最终两份权重增加忽略例外，用 `.gitattributes` 保持部署文件字节；按路径选择最终交付内容，实验改动保留本地。
- 保持原部署清单与包内说明不变，明确最终版本冻结与 V100 P95 尚待验证是两个独立状态；本次不重新训练或宣称新的运行指标。
