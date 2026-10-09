# EdgeFall 本地工业验证工作台

本项目是冻结的 **EdgeFall 视觉七头基线** 外部的单机 Web 操作界面；没有修改原始模型、权重、CLI 推理入口及其评测协议。本版本不是生产级实时监控产品，也不部署至 Hugging Face Spaces。

## 快速安装与运行

推荐 Windows + WSL2 + Python 3.11 + 可用的 NVIDIA GPU：

```bash
git clone -b feature/local-fall-testing-ui https://github.com/kingmt123/detictive-fall-detection.git
cd detictive-fall-detection
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-ui.txt
python app.py
```

在 Windows 浏览器打开 http://127.0.0.1:7860。默认仅监听 `127.0.0.1`，不开启 Gradio 公共分享。原项目锁定的 CUDA/PyTorch 依赖不一定兼容每个系统；若首次安装失败，优先检查 NVIDIA 驱动、WSL2 GPU 映射、Python/PyTorch/CUDA 对应关系。

```bash
# 可选：未检测到 GPU 时测试 CPU（性能可能较差）
EDGEFALL_DEVICE=cpu python app.py
# 可选：指定端口
EDGEFALL_PORT=7861 python app.py
```

## 界面结构

1. **视频检测工作台**：上传视频、执行原七头模型推理、查看时间戳叠加的结果回放，可定位到运动峰值。
2. **事件与证据**：候选时间轴（开始/峰值/结束），从原视频直接提取「开始前 / 峰值 / 结束后」三帧证据；不虚构人体检测框或逐帧跌倒标签。
3. **模型诊断**：视频级融合评分、七头原始 logits 分布、各阶段耗时及 JSON 导出。
4. **历史与人工复核**：最近 100 条检测记录、人工确认/排除/无法判断、备注记录及复核 JSON 导出。

## 数据语义和安全边界

- `score` = `sigmoid(mean(member_logits))`：**视频级、未经校准的评分**，不是跌倒概率或工业安全报警阈值。
- `proposal` 仅是姿态运动算法提出的候选事件：不是已证实的跌倒事件，也不是逐帧分类器输出。
- 人工复核保存在独立的 `review.json`；**永不修改**冻结模型的 `result.json`。
- `latency_ms.per_frame` 是整段总耗时均值，不等于在线报警延迟或逐帧 P95。
- 当前没有实时摄像头流接入、分人连续在线判定、告警闭环、设备联锁、远程身份认证、并发多租户、自动删除/加密、事件级精度证明，不应直接用于无人值守工业安全控制或医疗紧急报警。
- Gradio 界面默认开放给访问本机端口的人；本地操作员身份目前未验证，`review.json` 中明确记录 `local-unverified`。敏感影像应按使用机构的数据留存规则管理。

## 本地文件

全部保存在 `outputs/local_ui/run-*/`，并被 `.gitignore` 忽略：

- `result.json`：模型推理原样数据。
- `preview.mp4`：画面叠加视频级评分和姿态运动候选窗口。
- `evidence.jpg`：三帧真实视频截图。
- `case.json`：本地时间、文件名、推理设备和协议。
- `review.json`：人工复核事件及备注，仅复核后出现。

上传的视频不是永久保存的源片；历史记录打开后展示的是结果回放和事件截图，而不是伪装成原始输入的视频。

限制：上传视频 ≤250MB、长度 ≤180 秒；暂不支持在网页上对大批量或实时多路摄像头进行监控。

## 验证

```bash
python -m pytest tests/test_local_ui.py tests/test_local_ui_dashboard.py -q
```

这些测试使用合成视频和模拟推理模型，检验 UI、证据文件、历史/复核语义与路径保护。**无法替代**真实模型 GPU 运行、设备长时间稳定性、误报率、漏报率和报警延迟验证。
