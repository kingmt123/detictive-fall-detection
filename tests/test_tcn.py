"""models/tcn.py 冒烟测试：形状、参数量预算、因果性、GPU 时延。"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from models.tcn import FallTCN, count_params


def test_output_shape():
    m = FallTCN()
    x = torch.randn(8, 16, 17, 3)
    out = m(x)
    assert out.shape == (8,)
    out2 = m(torch.randn(8, 16, 51))
    assert out2.shape == (8,)


def test_param_budget():
    n = count_params(FallTCN())
    print(f"  FallTCN params: {n/1e6:.3f}M")
    assert n < 0.5e6, f"TCN 超出 0.5M 参数预算: {n}"


def test_causality():
    # 改动未来帧不应影响过去时刻的输出：用中间截断验证
    m = FallTCN().eval()
    x = torch.randn(1, 16, 51)
    with torch.no_grad():
        out_full = m(x)
        out_trunc = m(x[:, :8].clone())  # 前 8 帧
    # 截断输入的输出对应第 8 帧，无法直接比较 logit（取最后时刻），
    # 改为验证：篡改最后 4 帧，前 12 帧的隐状态应不变
    x2 = x.clone()
    x2[:, 12:] = 999.0
    with torch.no_grad():
        h1 = m.tcn(x.transpose(1, 2))[:, :, :12]
        h2 = m.tcn(x2.transpose(1, 2))[:, :, :12]
    assert torch.allclose(h1, h2, atol=1e-5), "TCN 违反因果性"


def test_latency_budget():
    m = FallTCN().eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = m.to(device)
    x = torch.randn(1, 16, 51, device=device)
    with torch.no_grad():
        for _ in range(20):
            m(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            m(x)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 200 * 1000
    print(f"  FallTCN 单窗推理 [{device}]: {dt:.2f}ms")
    assert dt < 5.0, f"TCN 时延超预算: {dt:.2f}ms"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
