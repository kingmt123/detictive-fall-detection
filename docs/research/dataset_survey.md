# 跌倒检测数据集快速盘点

> 竞赛项目：视觉实时跌倒检测 | 初赛截止：2026-09-01 | 调研日期：2026-08-16
> 状态：🚧 撰写中（逐节追加）

## 目录
1. [OmniFall (HuggingFace)](#1-omnifall)
2. [Kaggle fall-video-dataset (payutch)](#2-kaggle-fall-video-dataset)
3. [UR Fall Detection](#3-ur-fall-detection)
4. [Le2i Fall Detection](#4-le2i)
5. [统一标注格式可行性分析（YOLO检测框 + 片段级事件标注）](#5-格式转换可行性)
6. [16天周期推荐方案](#6-推荐方案)

---

## 1. OmniFall
- **地址**：https://huggingface.co/datasets/simplexsigil2/omnifall ｜ 论文：arXiv:2505.19889 ｜ 代码：github.com/simplexsigil/omnifall
- **定位**：2025年发布的统一跌倒检测基准，三个互补子集共享**16类活动标签体系**，**密集时间段标注（dense temporal segment annotations，帧级粒度）**——天然契合"片段级跌倒事件标注"需求。
- **三子集构成**：
  - **OF-Staged**：整合 **8个公开实验室摆拍数据集**（含 cmdfall、UP-Fall、Le2i、OOPS 等来源，具体8个清单见论文附录 A.3，待验证完整名单），带跨被试/跨视角划分。
  - **OF-In-the-Wild**：来自 OOPS 的**真实意外事故视频**，**仅作测试集**（held-out test-only），用于评估真实场景泛化。
  - **OF-Synthetic**：**12,000 段扩散模型生成的合成视频**，覆盖 staged 数据缺乏的人口学多样性（老人、儿童等）。
- **规模**：全基准 **33 小时密集标注素材**（论文摘要另提到约80小时口径，待验证具体定义）；16类活动统一标签。
- **标注格式**：帧级/片段级时间标注 + 16类活动分类（含 fall / lying 等关键类）。**明确有片段级时间段标注**，可直接映射为 `(clip_id, t_start, t_end)`。无现成 YOLO 检测框，需自行用检测器生成。
- **下载方式**：`huggingface_hub` / `hf CLI`（`hf download simplexsigil2/omnifall --repo-type dataset`）。HF 页面 API 访问本次异常（返回空，待验证是否有 gated 限制）。
- **许可**：**CC BY-SA 4.0**（注意 SA 传染性，竞赛自用可，商用需谨慎）。
- **评价**：⭐ 时间紧任务下的首选——标注已统一、有真实场景测试集、HF 一键下载。主要风险：33h 全量下载体积（待验证）与下载速度。

> **⚠️ 重大更正（2026-08-16 实测）**：OmniFall HF 仓库**只是标注层，不含源数据集视频**（parquet 仅百余 KB 的标注/元数据）。唯一可直接下载的视频是 **OF-Syn 合成集**（`data_files/omnifall-synthetic_av1.tar`，实测 9.7GB，12,000 段带密集标注的合成视频）。OF-Staged 8 源（CMDFall/UP-Fall/Le2i/CAUCAFall/GMDCSA24/EDF/OCCU/MCFD）与 OF-ItW(OOPS) 的**视频需到各原始来源另行获取**，OmniFall 的 labels/splits 直接复用。16 类标签含 fall/fallen/sit_down/sitting/lie_down/lying 等易混类，正是本赛题的核心判别对象。

## 2. Kaggle fall-video-dataset (payutch)
- **地址**：https://www.kaggle.com/datasets/payutch/fall-video-dataset
- **内容**：第三方**汇编数据集**，合并三个公开来源：
  1. Harvard Dataverse "Fall Vision" 基准视频集（doi:10.7910/DVN/75QPKK）
  2. Figshare "2017 activities / 29 subjects" 视频跌倒数据集
  3. 蒙特利尔大学 **Multiple Cameras Fall Dataset**（24个场景 × 8个IP相机，前22场景含跌倒+干扰事件，后2场景仅干扰）
- **规模**：整包 zip **约16 GB**（16,070,720,654 bytes）；约2.6万次下载，热度高。各源内部视频数待验证。
- **标注**：**随原始来源而异，无统一标注格式**——这是最大短板。Multiple Cameras 源有场景级描述；其余来源标注情况待验证。基本可判定**无现成片段级 (t_start, t_end) 标注**，需人工或半自动补标。
- **许可**：CC0 Public Domain（但注意：上游来源各自许可需单独核查，汇编者标 CC0 不代表上游无限制，待验证）。
- **下载**：Kaggle 页面下载需登录；推荐 `kaggle datasets download -d payutch/fall-video-dataset`（需 API key）。
- **评价**：体量大、免费、监控视角多，但标注不统一、16GB 下载重，16天周期内性价比偏低。

## 3. UR Fall Detection (URFD)
- **地址**：http://fenix.ur.edu.pl/~mkepski/ds/uf.html （热舒夫大学）
- **规模**：**70 个序列 = 30 跌倒 + 40 ADL**；跌倒由**2台 Kinect**（cam0 平视 + cam1 顶视）双视角录制 → 60段跌倒视频 + 40段ADL视频。
- **模态**：**深度图序列（PNG16，含深度换算公式）+ RGB 帧序列 + 加速度计数据**（PS Move 60Hz / x-IMU 256Hz，含同步 CSV）。
- **标注**：帧级同步数据含时间戳；跌倒时间段需按序列推断（序列即单一事件，标注成本低）。
- **许可**：**CC BY-NC-SA 4.0，仅限非商业学术用途**（竞赛可用，注意非商业限制）。需引用 Kwolek & Kępski 2014 论文。
- **下载**：官网逐文件 zip 直接 HTTP 下载，无需注册，可脚本批量抓取。
- **评价**：经典小基准，适合做快速 baseline 验证与双视角消融；规模太小不足以单独支撑训练。

## 4. Le2i
- **地址**：官网 le2i.cnrs.fr（本次访问无响应，待验证；可用镜像 github.com/YifeiYang210/Fall_Detection_dataset）
- **规模**：**4个场景共 221 段视频**：Home(60) / Coffee room(70) / Office(64) / Lecture room(27)。
- **模态**：RGB 视频，320×240 @ 25FPS，单人，画质较高（另有深度/加速度模态的说法待验证——经典 Le2i 以 RGB 为主）。
- **标注**：**仅 Home 与 Coffee room 两个子集有 Annotation_files，明确给出跌倒开始/结束帧号**——可直接转 `(clip_id, t_start, t_end)`；Office/Lecture room 标注缺失（待验证是否有社区补全版本）。
- **许可/下载**：官网需申请；GitHub 有多个镜像与 Roboflow 加工版（3010张检测框标注图像，可应急当 YOLO 数据）。许可条款待验证，学术使用为主。
- **评价**：自带片段级标注（部分场景），与目标格式吻合度最高的传统数据集；分辨率低、场景少。

## 5. 格式转换可行性
目标统一格式：**YOLO 检测框（帧级 person bbox）+ 片段级跌倒事件标注 `(clip_id, t_start, t_end)`**。

| 数据集 | 片段级事件标注转换 | YOLO 框转换 | 总体可行性 |
|---|---|---|---|
| **OmniFall** | ✅ 原生密集时间段标注，映射 fall/lying 类即可得 (t_start, t_end)，成本极低 | ⚠️ 无框标注，用预训练 YOLOv8/11 自动打框+抽检，1天内可完成 | ⭐⭐⭐⭐⭐ 最高 |
| **Kaggle payutch** | ❌ 标注不统一，大部分需人工补标跌倒时间段，16天周期下风险大 | ⚠️ 同需自动打框；Roboflow 上有相关加工版可部分借用 | ⭐⭐ 偏低 |
| **URFD** | ✅ 每个序列=单一事件，跌倒段可按加速度计峰值+人工抽检快速圈定（半天） | ⚠️ PNG 帧序列可直接喂 YOLO 自动打框 | ⭐⭐⭐⭐ 高（量小） |
| **Le2i** | ✅ Home/Coffee 子集自带起止帧号，脚本直转；另两场景缺失（待验证社区补全） | ⚠️ Roboflow 已有 3010 张 YOLO 格式框图可直接用 | ⭐⭐⭐⭐ 高 |

**通用转换管线（建议）**：
1. 抽帧 → 预训练 YOLOv8 生成 person 检测框（`tracker` 维持 ID）→ 存 YOLO txt；
2. 事件标注统一成 CSV：`clip_id, t_start, t_end, label(fall/adl)`；
3. 对无时间段标注的源：先用加速度计/姿态突变启发式预圈 → 人工确认（仅 URFD 量级可行，Kaggle 16GB 放弃人工）。

## 6. 推荐方案
**16天周期（截至 2026-09-01 初赛）推荐采用 3+1 组合：**

1. **OmniFall（主力，必选）** — 唯一自带统一片段级标注、且含真实场景测试集（OF-ItW）与1.2万合成视频（OF-Syn）的数据集；HF 一键下载，标注转换成本最低。Day 1-2 下载+转格式，Day 3 即可开训。许可 CC BY-SA 4.0 竞赛可接受。
2. **URFD（快速 baseline 验证）** — 70 序列小体量，半天即可跑通"下载→抽帧→打框→事件标注"全管线，用于 Day 1-3 验证 pipeline 正确性，避免在大数据上踩坑。免注册直接下载。
3. **Le2i（补充训练多样性）** — Home/Coffee 子集自带起止帧标注，Roboflow 版还提供现成 YOLO 框，双份红利；4场景补充 OmniFall 未覆盖的室内布局。
4. **Kaggle payutch（备选，仅有余力时启用）** — 16GB、标注不统一，仅当 pipeline 提前跑通且需扩量时，抽取其中 Multiple Cameras 多视角子集做域泛化增强；否则放弃。

**时间分配建议**：D1-2 OmniFall 下载+格式转换；D3-4 URFD/Le2i 管线验证；D5-10 训练+迭代；D11-14 OF-ItW 真实场景评测调优；D15-16 提交缓冲。

---
> **待验证清单**：OmniFall 是否 gated 及确切下载体积、OF-Staged 完整8源名单、Kaggle 各上游许可、Le2i 官网当前可用性与其深度/加速度模态说法、Office/Lecture 标注社区补全版。
