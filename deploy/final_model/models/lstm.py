"""因果 LSTM 跌倒分类基线。

该模块借鉴 pose-based-realtime-fall-detection 的特征组织：每帧 17 个
关键点的 ``x, y, conf`` 与相邻帧 ``dx, dy``。它使用当前项目已经冻结的
51 维归一化姿态，而不改变训练/验证集、窗口长度或标签语义。
"""
import torch
from torch import nn

JOINT_COUNT = 17
POSE_DIM = JOINT_COUNT * 3
VELOCITY_DIM = JOINT_COUNT * 2
FEATURE_DIM = POSE_DIM + VELOCITY_DIM
POSE51_FEATURES = "pose51"
POSE51_VELOCITY34_FEATURES = "pose51_velocity34"
SUPPORTED_INPUT_FEATURES = (POSE51_FEATURES, POSE51_VELOCITY34_FEATURES)


def _as_joint_pose(x: torch.Tensor) -> torch.Tensor:
    """规范化输入为 ``(B,T,17,3)`` 的浮点姿态张量。"""
    if x.ndim == 4:
        if x.shape[-2:] != (JOINT_COUNT, 3):
            raise ValueError("四维输入必须为 (B, T, 17, 3)")
        pose = x
    elif x.ndim == 3:
        if x.shape[-1] != POSE_DIM:
            raise ValueError("三维输入最后一维必须为 51")
        pose = x.reshape(*x.shape[:2], JOINT_COUNT, 3)
    else:
        raise ValueError("输入必须为 (B, T, 17, 3) 或 (B, T, 51)")
    if not torch.is_floating_point(pose):
        raise TypeError("姿态输入必须是浮点张量")
    return pose


def feature_dimension(input_features: str) -> int:
    """返回已注册姿态特征协议的每帧维度。"""
    if input_features == POSE51_FEATURES:
        return POSE_DIM
    if input_features == POSE51_VELOCITY34_FEATURES:
        return FEATURE_DIM
    raise ValueError(f"未知 input_features: {input_features!r}")


def pose51_features(x: torch.Tensor) -> torch.Tensor:
    """规范化并展平为现有冻结缓存所使用的 51 维姿态。"""
    return _as_joint_pose(x).flatten(2)


def add_velocity_features(x: torch.Tensor) -> torch.Tensor:
    """将 ``(B,T,17,3)`` 或 ``(B,T,51)`` 姿态扩展为 ``(B,T,85)``。

    速度是当前帧与前一帧的 ``x/y`` 差；第一个时间步以及相邻任一帧无
    关键点置信度时的速度均为零，避免姿态重新出现时形成伪运动。
    """
    pose = _as_joint_pose(x)

    velocity = torch.zeros_like(pose[..., :2])
    if pose.shape[1] > 1:
        valid_pair = (pose[:, 1:, :, 2] > 0) & (pose[:, :-1, :, 2] > 0)
        delta = pose[:, 1:, :, :2] - pose[:, :-1, :, :2]
        velocity[:, 1:] = delta * valid_pair.unsqueeze(-1).to(delta.dtype)
    return torch.cat((pose.flatten(2), velocity.flatten(2)), dim=-1)


class FallLSTM(nn.Module):
    """以最后时间步分类的单向 LSTM：姿态窗口 → 跌倒 logit。"""

    def __init__(
        self,
        *,
        input_dim: int = FEATURE_DIM,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        input_features: str = POSE51_VELOCITY34_FEATURES,
    ) -> None:
        super().__init__()
        expected_dim = feature_dimension(input_features)
        if input_dim != expected_dim:
            raise ValueError(f"input_dim 必须为 {input_features} 的维度 {expected_dim}")
        if hidden_size < 1 or num_layers < 1:
            raise ValueError("hidden_size 和 num_layers 必须为正整数")
        if not 0 <= dropout < 1:
            raise ValueError("dropout 必须位于 [0, 1)")
        self.input_features = input_features
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回每个窗口的 sigmoid 前二分类 logit，形状为 ``(B,)``。"""
        features = (
            add_velocity_features(x)
            if self.input_features == POSE51_VELOCITY34_FEATURES
            else pose51_features(x)
        )
        sequence, _ = self.lstm(features)
        return self.head(sequence[:, -1]).squeeze(-1)


def count_params(model: nn.Module) -> int:
    """返回可训练参数总数。"""
    return sum(parameter.numel() for parameter in model.parameters())
