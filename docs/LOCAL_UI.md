# EdgeFall 本地视频测试界面

这是在原七头模型外增加的 **本地测试功能**。不会修改 `tools.infer_seven_head`、模型结构、权重或既有评分方式，也不部署 Hugging Face Spaces。

## 安装和启动

在仓库根目录使用 Python 3.11 与适配本机驱动的 PyTorch 环境，安装原项目依赖及界面依赖：

```bash
pip install -r requirements.txt
pip install -r requirements-ui.txt
python app.py
```

打开 http://127.0.0.1:7860，上传视频并点击「开始检测」。默认优先选择可用 CUDA，否则使用 CPU（CPU 性能尚未验证）。在 Windows + WSL2 下，请在 WSL2 里启动并通过 Windows 浏览器访问该地址。无需 Hugging Face Spaces。

仓库内 `requirements.txt` 锁定的 CUDA 版本不一定适配所有系统。首次安装若发生包版本冲突，应先检查 Python、CUDA 与 PyTorch 的兼容性，不要为解决 UI 安装而擅自更改原权重。

## 输出

视频级七头评分（不是校准后的真实概率）和各成员 raw logits；从姿态运动提出的候选区间、运动峰值及 track ID；分阶段耗时；可下载 JSON 与标注运动区间的 MP4。不会伪造人体检测框或逐帧跌倒标签。

结果保存到 `outputs/local_ui/run-*/`。上传视频限制 250 MB 和 180 秒。若缺少 ffmpeg，会退回到 MP4V 编码，部分浏览器可能无法预览。

## 验证

```bash
python -m pytest tests/test_local_ui.py -q
```

此验证使用合成视频与模拟模型，并非原权重 GPU 推理或模型准确率复测。
