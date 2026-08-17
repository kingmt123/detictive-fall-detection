# 项目交接速查

> 最后更新：2026-08-17 17:00
> master HEAD：`e9496d7`
> GitHub：https://github.com/kingmt123/detictive-fall-detection

## 一句话

YOLO11n-pose + 轻量跟踪 + 物理规则 + 事件聚合的纯视觉跌倒检测方案，deadline 2026-09-01。

## 快速启动

```bash
git clone https://github.com/kingmt123/detictive-fall-detection.git
cd detictive-fall-detection
pip install -r requirements.txt   # CUDA PyTorch 需要单独安装
python -m pytest -q               # 应该 121 passed
```

## 已完成

| 里程碑 | 测试数 | 提交 |
|---|---|---|
| Round 0 确定性规则基线 | 74 | `efdf799` |
| Round 1 可复用推理引擎 + URFD val | 86 | `bd84fbc` |
| Gate 2A 统一视频源 + pose cache | 118 | `a52f37d` |
| Gate 2B rule evaluator tar URI | 121 | `e9496d7` |
| 20-clip canary | — | 完成，外推见下 |

## URFD val 基线

14/14 clips，P@R90=0.80，P@R95=0.80，MAP=80.0%，FP=2（`adl-12-cam0`、`adl-40-cam0`）。

## 20-clip canary 外推

- 1,200 OF-Syn val：~59 min，~47.6MB
- 10,800 train：~9.1h，~429MB
- D: 剩余 83GB，余量 ≥30%

## 下一步

1. 实现 TCN window consumer contract（用 canary 的 20 个 NPZ）
2. 全量 cache（train 10,800 + val 1,200）
3. FallTCN 训练（只读 cache）
4. val 消融 + 阈值选择
5. 1080P/V100 + test seal + 匿名提交

## 关键约束

- test split 不得使用
- C 盘仅剩 5.8GB，临时文件必须在 D:
- OF-Syn archive 9.7GB，tar handle 必须批次复用
- 模型 ≤20M 参数，FP32 ≤80MB，推理 ≤100ms
- 材料必须匿名

## 详细文档

- `HANDOVER.md` — 完整技术交接
- `DEVLOG.md` — 开发决策日志
- `README.md` — 项目说明和命令
- `.hermes/plans/` — 执行计划和门禁
