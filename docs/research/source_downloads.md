# 跌倒检测源数据集下载进度报告

> 目标：为视觉实时跌倒检测竞赛获取免注册源数据集视频。总量上限 15GB，单文件 20 分钟未完成即放弃。
> 最后更新：2026-08-16 15:57（进行中）

## 状态总览

| 数据集 | 状态 | 说明 |
|---|---|---|
| URFD | 🔄 下载中 | 官网可访问，页面提供单文件 mp4 直链 |
| Le2i | ⏳ 待处理 | 计划用 GitHub 镜像 |
| UP-Fall | ⏳ 待处理 | 寻找官方免注册入口 |
| CMDFall / CAUCAFall / GMDCSA24 / EDF / OCCU / MCFD / OOPS | ⏳ 仅记录获取方式 | 需注册的只记录不下载 |

---

## 1. URFD (UR Fall Detection Dataset)

- 官网：http://fenix.ur.edu.pl/~mkepski/ds/uf.html （可访问 ✅）
- 论文：Kwolek & Kepski, CMBP 2014（http://home.agh.edu.pl/~bkw/research/pdf/2014/KwolekKepski_CMBP2014.pdf）
- 页面结构（实际探测）：`/ds/data/` 下按单文件提供：
  - `adl-01..40-cam0.mp4` + `adl-XX-acc.csv`（40 个日常活动序列，顶视 cam0）
  - `fall-01..30-cam1.mp4` + `fall-XX-acc.csv`（30 个跌倒序列，前视 cam1）
  - 老版整包 zip（urfall-cam0-rgb.zip 等）待探测是否仍有效
- 许可：学术研究免费使用（官网声明 free for research use）
- 下载目录：`data/sources/urfd/`

（下载结果待补充）

---

## 主线接管记录（2026-08-16 16:10，Agent 超时后）

- **URFD**：已确认单文件直链模式 `http://fenix.ur.edu.pl/~mkepski/ds/data/{fall|adl}-XX-camY.mp4`，主线用后台 curl 循环批量下载中（fall-01..30 × cam0/cam1 + adl-01..40 × cam0 + 加速度 CSV）。
- **Le2i**：GitHub 搜索无可用视频镜像（YifeiYang210 仓库仅 59KB 文档）；官网需申请。HF 上仅有 `seanphan/le2i-sentinel-frames`（抽帧版）。**结论：暂缓**，OF-Syn+URFD 够用再回来补。
- **HF 候选数据集（待 D2 评估适配性）**：
  - `Nshisei/multimodal_multiangle_fall_detection_dataset`（多视角多模态，287 下载）
  - `Simuletic/CCTV_Incident_Dataset_Fall_Lying_Down_Detection`（CCTV 风格，206 下载）
  - `mervinpraison/upfall-detection-actual`（疑似 UP-Fall 子集）
  - `DeZan/fall-detection`（107 下载）
- **Kaggle payutch**：无 API key（~/.kaggle 不存在），且 16GB 标注不统一，维持"备选放弃"判断。
- **OF-Syn**：9.7GB tar 后台断点续传下载中（data/omnifall/data_files/）。
- **教训**：下载类任务不再派 Agent（600s 超时限制 + 慢速站点易卡死），改由主线 `terminal(background=true)` 直接跑 curl 循环。
