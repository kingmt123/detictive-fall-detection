"""预算受控的四流多尺度因果 TCN 与时序注意力跌倒分类器。"""

from __future__ import annotations

import math

import torch
from torch import nn

from models.tcn_multistream import COCO_BONE_EDGES

JOINT_DIM = 17 * 3
BONE_DIM = len(COCO_BONE_EDGES) * 2
MOTION_DIM = 17 * 2
MOTION_VALIDITY_DIM = MOTION_DIM + 17
ACCEL_DIM = 17 * 2
GEOMETRY_DIM = 6
RULE_DIM = 12
PIFR_DIM = 9
TRANSITION_RULE_DIM = 8
# Continuous, causal cues used to distinguish an uncontrolled transition from
# controlled lie-down / stand-up motion.  These are model inputs, never hard
# alarm rules.
KINEMATIC_RULE_DIM = 15
STREAM_DIMENSIONS = (JOINT_DIM, BONE_DIM, MOTION_DIM, ACCEL_DIM)
GEOMETRY_STREAM_DIMENSIONS = (JOINT_DIM, BONE_DIM, MOTION_DIM, GEOMETRY_DIM)
GEOMETRY_RULE_STREAM_DIMENSIONS = GEOMETRY_STREAM_DIMENSIONS + (RULE_DIM,)
GEOMETRY_PIFR_STREAM_DIMENSIONS = GEOMETRY_STREAM_DIMENSIONS + (PIFR_DIM,)
GEOMETRY_RULE_PIFR_STREAM_DIMENSIONS = GEOMETRY_STREAM_DIMENSIONS + (
    RULE_DIM + PIFR_DIM,
)


def _coco_adjacency() -> torch.Tensor:
    """Return the fixed, symmetrically normalized 17-joint COCO graph."""
    adjacency = torch.eye(17, dtype=torch.float32)
    for left, right in COCO_BONE_EDGES:
        adjacency[left, right] = 1.0
        adjacency[right, left] = 1.0
    degree = adjacency.sum(dim=1).clamp_min(1.0)
    return adjacency / degree.sqrt().unsqueeze(1) / degree.sqrt().unsqueeze(0)


def _as_pose(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4 and x.shape[-2:] == (17, 3):
        pose = x
    elif x.ndim == 3 and x.shape[-1] == JOINT_DIM:
        pose = x.reshape(*x.shape[:2], 17, 3)
    else:
        raise ValueError("输入必须是 (B,T,17,3) 或 (B,T,51)")
    if not torch.is_floating_point(pose):
        raise TypeError("姿态必须是浮点张量")
    return pose


def build_four_stream_features(
    x: torch.Tensor,
    *,
    use_geometry: bool = False,
    motion_validity_mask: bool = False,
    per_joint_mask_correction: bool = False,
) -> tuple[torch.Tensor, ...]:
    """派生 joint/bone/motion 与 accel 或 bbox/reliability 第四流。

    16 条 COCO 骨骼边产生 32D，而不是 34D。速度需相邻帧均可见；加速度
    需连续三帧可见，首帧/次帧或姿态缺失时统一置零。
    """
    if use_geometry:
        if x.ndim != 3 or x.shape[-1] != JOINT_DIM + GEOMETRY_DIM:
            raise ValueError("Geometry 模式输入必须是 (B,T,57)")
        geometry = x[..., JOINT_DIM:]
        pose = _as_pose(x[..., :JOINT_DIM])
    else:
        geometry = None
        pose = _as_pose(x)
    coordinates = pose[..., :2]
    joint = pose.flatten(2)
    observed = torch.any(pose[..., 2] > 0, dim=-1)
    edges = torch.as_tensor(COCO_BONE_EDGES, device=pose.device)
    bone = coordinates[..., edges[:, 1], :] - coordinates[..., edges[:, 0], :]
    bone_observed = (
        (pose[..., edges[:, 0], 2] > 0) & (pose[..., edges[:, 1], 2] > 0)
        if per_joint_mask_correction
        else observed.unsqueeze(-1)
    )
    bone = bone * bone_observed.unsqueeze(-1).to(bone.dtype)

    motion = torch.zeros_like(coordinates)
    motion_validity = torch.zeros(
        (*pose.shape[:2], 17), dtype=pose.dtype, device=pose.device
    )
    if pose.shape[1] > 1:
        joint_pair_observed = (pose[:, 1:, :, 2] > 0) & (pose[:, :-1, :, 2] > 0)
        pair_observed = observed[:, 1:] & observed[:, :-1]
        motion_observed = joint_pair_observed if per_joint_mask_correction else pair_observed.unsqueeze(-1)
        motion[:, 1:] = (coordinates[:, 1:] - coordinates[:, :-1]) * motion_observed.unsqueeze(-1).to(coordinates.dtype)
        motion_validity[:, 1:] = joint_pair_observed.to(pose.dtype)
    accel = torch.zeros_like(coordinates)
    if pose.shape[1] > 2:
        triple_observed = observed[:, 2:] & observed[:, 1:-1] & observed[:, :-2]
        accel[:, 2:] = (motion[:, 2:] - motion[:, 1:-1]) * (
            triple_observed.unsqueeze(-1).unsqueeze(-1).to(coordinates.dtype)
        )
    if geometry is not None:
        motion_stream = (
            torch.cat((motion.flatten(2), motion_validity), dim=-1)
            if motion_validity_mask
            else motion.flatten(2)
        )
        return joint, bone.flatten(2), motion_stream, geometry
    motion_stream = (
        torch.cat((motion.flatten(2), motion_validity), dim=-1)
        if motion_validity_mask
        else motion.flatten(2)
    )
    return joint, bone.flatten(2), motion_stream, accel.flatten(2)


def _masked_joint_center(
    pose: torch.Tensor, indices: tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an x/y center and visibility for a small joint group."""
    selected = pose[..., indices, :]
    visible = selected[..., 2] > 0
    weights = visible.to(pose.dtype).unsqueeze(-1)
    count = weights.sum(dim=-2)
    center = (selected[..., :2] * weights).sum(dim=-2) / count.clamp_min(1.0)
    return center, count.squeeze(-1) > 0


def build_rule_features(x: torch.Tensor) -> torch.Tensor:
    """Build 12 causal, scale-normalized pose rules for a learnable rule stream.

    The values are deliberately continuous rather than thresholds.  They encode
    posture (torso orientation and joint-height relations), motion, bbox motion,
    and pose confidence; the downstream encoder learns how to weight them.
    """
    if x.ndim != 3 or x.shape[-1] != JOINT_DIM + GEOMETRY_DIM:
        raise ValueError("规则流输入必须是 (B,T,57) Geometry 特征")
    pose = _as_pose(x[..., :JOINT_DIM])
    geometry = x[..., JOINT_DIM:]
    coordinates = pose[..., :2]
    confidence = pose[..., 2]
    shoulders, shoulders_visible = _masked_joint_center(pose, (5, 6))
    hips, hips_visible = _masked_joint_center(pose, (11, 12))
    knees, knees_visible = _masked_joint_center(pose, (13, 14))
    ankles, ankles_visible = _masked_joint_center(pose, (15, 16))
    head, head_visible = _masked_joint_center(pose, (0,))
    torso = shoulders - hips
    torso_visible = shoulders_visible & hips_visible
    torso_length = torch.linalg.vector_norm(torso, dim=-1)
    horizontalness = torso[..., 0].abs() / torso_length.clamp_min(1e-4)

    joint_visible = confidence > 0
    y_values = coordinates[..., 1]
    masked_min = y_values.masked_fill(~joint_visible, torch.inf).amin(dim=-1)
    masked_max = y_values.masked_fill(~joint_visible, -torch.inf).amax(dim=-1)
    vertical_span = torch.where(
        joint_visible.any(dim=-1), masked_max - masked_min, torch.zeros_like(masked_max)
    )

    def relation(
        upper: torch.Tensor,
        upper_visible: torch.Tensor,
        lower: torch.Tensor,
        lower_visible: torch.Tensor,
    ) -> torch.Tensor:
        return (upper[..., 1] - lower[..., 1]) * (upper_visible & lower_visible).to(
            pose.dtype
        )

    pose_motion = torch.zeros_like(torso_length)
    torso_turn = torch.zeros_like(torso_length)
    bbox_center_velocity = torch.zeros_like(torso_length)
    bbox_height_velocity = torch.zeros_like(torso_length)
    bbox_aspect_velocity = torch.zeros_like(torso_length)
    if pose.shape[1] > 1:
        joint_pair = joint_visible[:, 1:] & joint_visible[:, :-1]
        displacement = torch.linalg.vector_norm(
            coordinates[:, 1:] - coordinates[:, :-1], dim=-1
        )
        denominator = joint_pair.sum(dim=-1).clamp_min(1).to(pose.dtype)
        pose_motion[:, 1:] = (displacement * joint_pair.to(pose.dtype)).sum(
            dim=-1
        ) / denominator
        torso_pair = torso_visible[:, 1:] & torso_visible[:, :-1]
        torso_turn[:, 1:] = (
            horizontalness[:, 1:] - horizontalness[:, :-1]
        ).abs() * torso_pair.to(pose.dtype)
        bbox_pair = (geometry[:, 1:, 5] > 0) & (geometry[:, :-1, 5] > 0)
        bbox_center_velocity[:, 1:] = (
            geometry[:, 1:, 1] - geometry[:, :-1, 1]
        ) * bbox_pair.to(pose.dtype)
        bbox_height_velocity[:, 1:] = (
            torch.log(geometry[:, 1:, 3].clamp_min(1e-4))
            - torch.log(geometry[:, :-1, 3].clamp_min(1e-4))
        ) * bbox_pair.to(pose.dtype)
        bbox_aspect_velocity[:, 1:] = (
            geometry[:, 1:, 4] - geometry[:, :-1, 4]
        ) * bbox_pair.to(pose.dtype)

    return torch.stack(
        (
            horizontalness * torso_visible.to(pose.dtype),
            torso_length * torso_visible.to(pose.dtype),
            relation(head, head_visible, hips, hips_visible),
            relation(hips, hips_visible, knees, knees_visible),
            relation(knees, knees_visible, ankles, ankles_visible),
            vertical_span,
            pose_motion,
            torso_turn,
            bbox_center_velocity,
            bbox_height_velocity,
            bbox_aspect_velocity,
            (confidence * joint_visible.to(pose.dtype)).sum(dim=-1)
            / joint_visible.sum(dim=-1).clamp_min(1).to(pose.dtype),
        ),
        dim=-1,
    )


def build_transition_rule_features(x: torch.Tensor) -> torch.Tensor:
    """Build eight causal signals that distinguish a fall transition from lying.

    Unlike the general posture rule stream, these features emphasize *change*: a
    body becoming horizontal, downward motion, shrinking vertical span, and
    bbox changes.  Static lying may be horizontal but lacks most of this
    transition evidence.  Missing or first-frame derivatives are zero.
    """
    if x.ndim != 3 or x.shape[-1] != JOINT_DIM + GEOMETRY_DIM:
        raise ValueError("转变规则输入必须是 (B,T,57) Geometry 特征")
    pose = _as_pose(x[..., :JOINT_DIM])
    geometry = x[..., JOINT_DIM:]
    coordinates = pose[..., :2]
    confidence = pose[..., 2]
    visible = confidence > 0
    shoulders, shoulders_visible = _masked_joint_center(pose, (5, 6))
    hips, hips_visible = _masked_joint_center(pose, (11, 12))
    torso = shoulders - hips
    torso_visible = shoulders_visible & hips_visible
    horizontalness = torso[..., 0].abs() / torch.linalg.vector_norm(
        torso, dim=-1
    ).clamp_min(1e-4)
    weights = visible.to(pose.dtype).unsqueeze(-1)
    centre = (coordinates * weights).sum(dim=-2) / weights.sum(dim=-2).clamp_min(1.0)
    y_values = coordinates[..., 1]
    min_y = y_values.masked_fill(~visible, torch.inf).amin(dim=-1)
    max_y = y_values.masked_fill(~visible, -torch.inf).amax(dim=-1)
    span = torch.where(visible.any(dim=-1), max_y - min_y, torch.zeros_like(max_y))

    torso_turn = torch.zeros_like(horizontalness)
    centre_down = torch.zeros_like(horizontalness)
    span_velocity = torch.zeros_like(horizontalness)
    bbox_center_velocity = torch.zeros_like(horizontalness)
    bbox_height_velocity = torch.zeros_like(horizontalness)
    bbox_aspect_velocity = torch.zeros_like(horizontalness)
    pair_visible_fraction = torch.zeros_like(horizontalness)
    if pose.shape[1] > 1:
        pair_visible = visible[:, 1:] & visible[:, :-1]
        frame_pair = visible[:, 1:].any(dim=-1) & visible[:, :-1].any(dim=-1)
        torso_pair = torso_visible[:, 1:] & torso_visible[:, :-1]
        torso_turn[:, 1:] = (
            horizontalness[:, 1:] - horizontalness[:, :-1]
        ).abs() * torso_pair.to(pose.dtype)
        centre_down[:, 1:] = (centre[:, 1:, 1] - centre[:, :-1, 1]) * frame_pair.to(
            pose.dtype
        )
        span_velocity[:, 1:] = (span[:, 1:] - span[:, :-1]) * frame_pair.to(pose.dtype)
        bbox_pair = (geometry[:, 1:, 5] > 0) & (geometry[:, :-1, 5] > 0)
        bbox_center_velocity[:, 1:] = (
            geometry[:, 1:, 1] - geometry[:, :-1, 1]
        ) * bbox_pair.to(pose.dtype)
        bbox_height_velocity[:, 1:] = (
            torch.log(geometry[:, 1:, 3].clamp_min(1e-4))
            - torch.log(geometry[:, :-1, 3].clamp_min(1e-4))
        ) * bbox_pair.to(pose.dtype)
        bbox_aspect_velocity[:, 1:] = (
            geometry[:, 1:, 4] - geometry[:, :-1, 4]
        ) * bbox_pair.to(pose.dtype)
        pair_visible_fraction[:, 1:] = pair_visible.to(pose.dtype).mean(dim=-1)
    return torch.stack(
        (
            horizontalness * torso_visible.to(pose.dtype),
            torso_turn,
            centre_down,
            span_velocity,
            bbox_center_velocity,
            bbox_height_velocity,
            bbox_aspect_velocity,
            pair_visible_fraction,
        ),
        dim=-1,
    )


def build_discriminative_kinematic_features(x: torch.Tensor) -> torch.Tensor:
    """Build causal continuous cues for fall/lie-down and fall/stand-up separation.

    The stream contains joint/bbox acceleration, signed downward direction,
    first-frame posture, arm spread, jerk, torso angular velocity, a causal
    peak-to-stop cue, and post-transition stillness. Every derivative is
    confidence-masked; no future frame or hard decision rule is used. The
    classifier therefore learns whether a cue matters for a clip.
    """
    if x.ndim != 3 or x.shape[-1] != JOINT_DIM + GEOMETRY_DIM:
        raise ValueError("运动学流输入必须是 (B,T,57) Geometry 特征")
    pose = _as_pose(x[..., :JOINT_DIM])
    geometry = x[..., JOINT_DIM:]
    coordinates, confidence = pose[..., :2], pose[..., 2]
    visible = confidence > 0
    weights = visible.to(pose.dtype).unsqueeze(-1)
    centre = (coordinates * weights).sum(dim=-2) / weights.sum(dim=-2).clamp_min(1.0)
    shoulders, shoulders_visible = _masked_joint_center(pose, (5, 6))
    hips, hips_visible = _masked_joint_center(pose, (11, 12))
    torso = shoulders - hips
    torso_visible = shoulders_visible & hips_visible
    torso_length = torch.linalg.vector_norm(torso, dim=-1).clamp_min(1e-4)
    horizontalness = torso[..., 0].abs() / torso_length
    y_values = coordinates[..., 1]
    min_y = y_values.masked_fill(~visible, torch.inf).amin(dim=-1)
    max_y = y_values.masked_fill(~visible, -torch.inf).amax(dim=-1)
    vertical_span = torch.where(
        visible.any(dim=-1), max_y - min_y, torch.zeros_like(max_y)
    )

    left_wrist, left_wrist_visible = _masked_joint_center(pose, (9,))
    right_wrist, right_wrist_visible = _masked_joint_center(pose, (10,))
    wrist_visible = left_wrist_visible & right_wrist_visible & shoulders_visible
    arm_spread = (
        torch.linalg.vector_norm(left_wrist - right_wrist, dim=-1) / torso_length
    )
    arm_spread = arm_spread * wrist_visible.to(pose.dtype)

    joint_velocity = torch.zeros_like(coordinates)
    joint_acceleration = torch.zeros_like(coordinates)
    joint_jerk = torch.zeros_like(coordinates)
    centre_velocity = torch.zeros_like(horizontalness)
    centre_acceleration = torch.zeros_like(horizontalness)
    bbox_velocity = torch.zeros_like(horizontalness)
    bbox_acceleration = torch.zeros_like(horizontalness)
    bbox_jerk = torch.zeros_like(horizontalness)
    arm_velocity = torch.zeros_like(horizontalness)
    torso_angular_velocity = torch.zeros_like(horizontalness)
    peak_to_stop = torch.zeros_like(horizontalness)
    post_transition_stillness = torch.zeros_like(horizontalness)
    mean_joint_speed = torch.zeros_like(horizontalness)
    if pose.shape[1] > 1:
        pairs = visible[:, 1:] & visible[:, :-1]
        joint_velocity[:, 1:] = (
            coordinates[:, 1:] - coordinates[:, :-1]
        ) * pairs.unsqueeze(-1).to(pose.dtype)
        mean_joint_speed[:, 1:] = torch.linalg.vector_norm(
            joint_velocity[:, 1:], dim=-1
        ).mean(dim=-1)
        frames = visible[:, 1:].any(dim=-1) & visible[:, :-1].any(dim=-1)
        centre_velocity[:, 1:] = (centre[:, 1:, 1] - centre[:, :-1, 1]) * frames.to(
            pose.dtype
        )
        bbox_pairs = (geometry[:, 1:, 5] > 0) & (geometry[:, :-1, 5] > 0)
        bbox_velocity[:, 1:] = (
            geometry[:, 1:, 1] - geometry[:, :-1, 1]
        ) * bbox_pairs.to(pose.dtype)
        arm_pairs = wrist_visible[:, 1:] & wrist_visible[:, :-1]
        arm_velocity[:, 1:] = (arm_spread[:, 1:] - arm_spread[:, :-1]) * arm_pairs.to(
            pose.dtype
        )
        torso_pairs = torso_visible[:, 1:] & torso_visible[:, :-1]
        torso_angular_velocity[:, 1:] = (
            horizontalness[:, 1:] - horizontalness[:, :-1]
        ).abs() * torso_pairs.to(pose.dtype)
    if pose.shape[1] > 2:
        triples = visible[:, 2:] & visible[:, 1:-1] & visible[:, :-2]
        joint_acceleration[:, 2:] = (
            joint_velocity[:, 2:] - joint_velocity[:, 1:-1]
        ) * triples.unsqueeze(-1).to(pose.dtype)
        frames = (
            visible[:, 2:].any(dim=-1)
            & visible[:, 1:-1].any(dim=-1)
            & visible[:, :-2].any(dim=-1)
        )
        centre_acceleration[:, 2:] = (
            centre_velocity[:, 2:] - centre_velocity[:, 1:-1]
        ) * frames.to(pose.dtype)
        bbox_frames = (
            (geometry[:, 2:, 5] > 0)
            & (geometry[:, 1:-1, 5] > 0)
            & (geometry[:, :-2, 5] > 0)
        )
        bbox_acceleration[:, 2:] = (
            bbox_velocity[:, 2:] - bbox_velocity[:, 1:-1]
        ) * bbox_frames.to(pose.dtype)
    if pose.shape[1] > 3:
        quadruples = (
            visible[:, 3:] & visible[:, 2:-1] & visible[:, 1:-2] & visible[:, :-3]
        )
        joint_jerk[:, 3:] = (
            joint_acceleration[:, 3:] - joint_acceleration[:, 2:-1]
        ) * quadruples.unsqueeze(-1).to(pose.dtype)
        bbox_frames = (
            (geometry[:, 3:, 5] > 0)
            & (geometry[:, 2:-1, 5] > 0)
            & (geometry[:, 1:-2, 5] > 0)
            & (geometry[:, :-3, 5] > 0)
        )
        bbox_jerk[:, 3:] = (
            bbox_acceleration[:, 3:] - bbox_acceleration[:, 2:-1]
        ) * bbox_frames.to(pose.dtype)

    # A fall often shows a speed peak followed by a rapid settling phase.  The
    # running maximum makes this causal; controlled descent may still produce
    # this pattern, so the network rather than a threshold decides its value.
    downward_speed = centre_velocity.clamp_min(0.0)
    previous_peak = torch.cummax(downward_speed, dim=1).values
    peak_to_stop = (previous_peak - downward_speed).clamp_min(0.0)
    peak_to_stop[:, 0] = 0.0
    # Horizontal and slow is evidence of a post-transition settled state, but
    # remains a soft continuous cue to avoid rejecting windows that begin late.
    post_transition_stillness = (
        horizontalness * torso_visible.to(pose.dtype) / (1.0 + mean_joint_speed)
    )

    initial_horizontalness = horizontalness[:, :1].expand_as(
        horizontalness
    ) * torso_visible[:, :1].to(pose.dtype)
    initial_aspect = geometry[:, :1, 4].expand_as(horizontalness) * (
        geometry[:, :1, 5] > 0
    ).to(pose.dtype)
    initial_span = vertical_span[:, :1].expand_as(horizontalness)
    return torch.stack(
        (
            torch.linalg.vector_norm(joint_acceleration, dim=-1).mean(dim=-1),
            centre_acceleration,
            bbox_acceleration,
            centre_velocity,
            bbox_velocity,
            initial_horizontalness,
            initial_aspect,
            initial_span,
            arm_spread,
            arm_velocity,
            torch.linalg.vector_norm(joint_jerk, dim=-1).mean(dim=-1),
            bbox_jerk,
            torso_angular_velocity,
            peak_to_stop,
            post_transition_stillness,
        ),
        dim=-1,
    )


def _angle_feature(
    first: torch.Tensor,
    vertex: torch.Tensor,
    second: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Return a stable [0, 1] angle with zero for an unobservable triplet."""
    left = first - vertex
    right = second - vertex
    denominator = torch.linalg.vector_norm(left, dim=-1) * torch.linalg.vector_norm(
        right, dim=-1
    )
    cosine = (left * right).sum(dim=-1) / denominator.clamp_min(1e-4)
    angle = torch.acos(cosine.clamp(-1.0, 1.0)) / torch.pi
    return angle * valid.to(angle.dtype)


def _line_horizontalness(
    start: torch.Tensor, end: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Return one for a horizontal line and zero for a vertical/missing line."""
    vector = end - start
    value = vector[..., 0].abs() / torch.linalg.vector_norm(vector, dim=-1).clamp_min(
        1e-4
    )
    return value * valid.to(value.dtype)


def build_pifr_features(x: torch.Tensor) -> torch.Tensor:
    """Build the nine continuous pose-geometry quantities inspired by PIFR.

    PIFR combines a body centre with torso/head/shoulder/hip/leg/whole-body
    angles.  We keep those quantities continuous and confidence-masked, rather
    than copying its SVM thresholds, so the causal stream encoder can learn a
    dataset-specific decision rule.  Feature order is centre (x,y), torso
    horizontalness, nose-shoulder angle, shoulder/hip horizontalness, left and
    right leg bend, and nose-ankle horizontalness.
    """
    if x.ndim != 3 or x.shape[-1] != JOINT_DIM + GEOMETRY_DIM:
        raise ValueError("PIFR 特征输入必须是 (B,T,57) Geometry 特征")
    pose = _as_pose(x[..., :JOINT_DIM])
    coordinates = pose[..., :2]
    joint_visible = pose[..., 2] > 0
    visible_weights = joint_visible.to(pose.dtype).unsqueeze(-1)
    visible_count = visible_weights.sum(dim=-2)
    centre = (coordinates * visible_weights).sum(dim=-2) / visible_count.clamp_min(1.0)
    centre = centre * (visible_count > 0).to(pose.dtype)

    nose, nose_visible = _masked_joint_center(pose, (0,))
    shoulders, shoulders_visible = _masked_joint_center(pose, (5, 6))
    hips, hips_visible = _masked_joint_center(pose, (11, 12))
    ankles, ankles_visible = _masked_joint_center(pose, (15, 16))
    left_hip, left_hip_visible = _masked_joint_center(pose, (11,))
    right_hip, right_hip_visible = _masked_joint_center(pose, (12,))
    left_knee, left_knee_visible = _masked_joint_center(pose, (13,))
    right_knee, right_knee_visible = _masked_joint_center(pose, (14,))
    left_ankle, left_ankle_visible = _masked_joint_center(pose, (15,))
    right_ankle, right_ankle_visible = _masked_joint_center(pose, (16,))

    return torch.stack(
        (
            centre[..., 0],
            centre[..., 1],
            _line_horizontalness(shoulders, hips, shoulders_visible & hips_visible),
            _angle_feature(
                nose, shoulders, hips, nose_visible & shoulders_visible & hips_visible
            ),
            _line_horizontalness(
                pose[..., 5, :2],
                pose[..., 6, :2],
                joint_visible[..., 5] & joint_visible[..., 6],
            ),
            _line_horizontalness(
                pose[..., 11, :2],
                pose[..., 12, :2],
                joint_visible[..., 11] & joint_visible[..., 12],
            ),
            _angle_feature(
                left_hip,
                left_knee,
                left_ankle,
                left_hip_visible & left_knee_visible & left_ankle_visible,
            ),
            _angle_feature(
                right_hip,
                right_knee,
                right_ankle,
                right_hip_visible & right_knee_visible & right_ankle_visible,
            ),
            _line_horizontalness(nose, ankles, nose_visible & ankles_visible),
        ),
        dim=-1,
    )


def build_stream_features(
    x: torch.Tensor,
    *,
    use_geometry: bool = False,
    use_rules: bool = False,
    use_pifr: bool = False,
    use_kinematics: bool = False,
    kinematic_feature_dim: int = KINEMATIC_RULE_DIM,
    motion_validity_mask: bool = False,
    per_joint_mask_correction: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Build base streams plus optional learned-rule/PIFR geometry stream."""
    if (use_rules or use_pifr or use_kinematics) and not use_geometry:
        raise ValueError("规则/PIFR/运动学流需要 Geometry+bbox/observed 输入")
    streams = build_four_stream_features(
        x,
        use_geometry=use_geometry,
        motion_validity_mask=motion_validity_mask,
        per_joint_mask_correction=per_joint_mask_correction,
    )
    extras = []
    if use_rules:
        extras.append(build_rule_features(x))
    if use_pifr:
        extras.append(build_pifr_features(x))
    if extras:
        streams = (*streams, torch.cat(extras, dim=-1))
    if not use_kinematics:
        return streams
    if kinematic_feature_dim not in {12, KINEMATIC_RULE_DIM}:
        raise ValueError("运动学特征维度必须为 12 或当前版本维度")
    # 12D is the immutable schema used by existing checkpoints. New models
    # receive the complete schema, while old artifacts remain loadable.
    return (
        *streams,
        build_discriminative_kinematic_features(x)[..., :kinematic_feature_dim],
    )


class CausalMultiScaleBlock(nn.Module):
    """三核深度卷积、可选 GLU 门控和 SE 的因果残差块。"""

    def __init__(
        self,
        channels: int,
        dilation: int,
        dropout: float,
        *,
        use_gated_conv: bool = False,
    ) -> None:
        super().__init__()
        self.paddings = tuple((kernel - 1) * dilation for kernel in (3, 5, 7))
        self.depthwise = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel, groups=channels, dilation=dilation)
            for kernel in (3, 5, 7)
        )
        self.mix = nn.Conv1d(channels * 3, channels, 1)
        self.gate_mix = nn.Conv1d(channels * 3, channels, 1) if use_gated_conv else None
        self.norm = nn.BatchNorm1d(channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        reduced = max(4, channels // 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, reduced, 1),
            nn.ReLU(),
            nn.Conv1d(reduced, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [
            convolution(nn.functional.pad(x, (padding, 0)))
            for convolution, padding in zip(self.depthwise, self.paddings, strict=True)
        ]
        residual = x
        stacked = torch.cat(branches, dim=1)
        fused = self.mix(stacked)
        if self.gate_mix is not None:
            # The gate reads only causally padded branch activations, so it
            # cannot leak future frames while filtering transient pose noise.
            fused = fused * torch.sigmoid(self.gate_mix(stacked))
        fused = self.dropout(self.activation(self.norm(fused)))
        return residual + fused * self.se(fused)


class StreamEncoder(nn.Module):
    """单流：投影 → 多尺度 TCN → 可选因果 LSTM → 因果注意力。"""

    def __init__(
        self,
        input_dim: int,
        channels: int,
        output_dim: int,
        dropout: float,
        recurrent_layers: int = 0,
        use_transformer_encoder: bool = False,
        use_gated_conv: bool = False,
    ) -> None:
        super().__init__()
        if recurrent_layers < 0:
            raise ValueError("recurrent_layers 不能为负")
        self.input_projection = nn.Conv1d(input_dim, channels, 1)
        self.blocks = nn.Sequential(
            *(
                CausalMultiScaleBlock(
                    channels, dilation, dropout, use_gated_conv=use_gated_conv
                )
                for dilation in (1, 2, 4, 8)
            )
        )
        self.recurrent = (
            nn.LSTM(
                input_size=channels,
                hidden_size=channels,
                num_layers=recurrent_layers,
                batch_first=True,
                dropout=dropout if recurrent_layers > 1 else 0.0,
            )
            if recurrent_layers
            else None
        )
        self.recurrent_norm = nn.LayerNorm(channels) if recurrent_layers else None
        self.attention = nn.MultiheadAttention(
            channels, num_heads=4, dropout=dropout, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(channels)
        self.transformer_feedforward = (
            nn.Sequential(
                nn.Linear(channels, channels * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(channels * 2, channels),
            )
            if use_transformer_encoder
            else None
        )
        self.transformer_norm = (
            nn.LayerNorm(channels) if use_transformer_encoder else None
        )
        self.output_projection = nn.Linear(channels, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temporal = self.blocks(self.input_projection(x.transpose(1, 2))).transpose(1, 2)
        if self.recurrent is not None:
            recurrent, _ = self.recurrent(temporal)
            temporal = self.recurrent_norm(temporal + recurrent)
        length = temporal.shape[1]
        causal_mask = torch.ones(
            length, length, dtype=torch.bool, device=temporal.device
        ).triu(diagonal=1)
        attended, _ = self.attention(
            temporal, temporal, temporal, attn_mask=causal_mask
        )
        temporal = self.attention_norm(temporal + attended)
        if self.transformer_feedforward is not None:
            temporal = self.transformer_norm(
                temporal + self.transformer_feedforward(temporal)
            )
        return self.output_projection(temporal[:, -1])


class GraphTemporalBlock(nn.Module):
    """Fixed-skeleton graph convolution followed by a causal temporal convolution."""

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.dilation = dilation
        self.spatial = nn.Conv2d(channels, channels, 1)
        self.temporal = nn.Conv2d(channels, channels, (3, 1), dilation=(dilation, 1))
        self.norm = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        graph = torch.einsum("bctv,vw->bctw", x, adjacency)
        graph = self.spatial(graph)
        temporal = self.temporal(nn.functional.pad(graph, (0, 0, 2 * self.dilation, 0)))
        return x + self.dropout(torch.relu(self.norm(temporal)))


class STGCNJointEncoder(nn.Module):
    """A compact causal ST-GCN replacement for the flattened joint stream."""

    def __init__(self, channels: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_projection = nn.Conv2d(3, channels, 1)
        self.register_buffer("adjacency", _coco_adjacency(), persistent=False)
        self.blocks = nn.ModuleList(
            GraphTemporalBlock(channels, dilation, dropout) for dilation in (1, 2, 4, 8)
        )
        self.attention = nn.MultiheadAttention(
            channels, num_heads=4, dropout=dropout, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(channels)
        self.output_projection = nn.Linear(channels, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != JOINT_DIM:
            raise ValueError("ST-GCN Joint 流输入必须是 (B,T,51)")
        pose = x.reshape(*x.shape[:2], 17, 3)
        temporal = self.input_projection(pose.permute(0, 3, 1, 2))
        for block in self.blocks:
            temporal = block(temporal, self.adjacency)
        temporal = temporal.mean(dim=-1).transpose(1, 2)
        length = temporal.shape[1]
        causal_mask = torch.ones(
            length, length, dtype=torch.bool, device=temporal.device
        ).triu(diagonal=1)
        attended, _ = self.attention(
            temporal, temporal, temporal, attn_mask=causal_mask
        )
        return self.output_projection(self.attention_norm(temporal + attended)[:, -1])


class ResidualSTGCNJointEncoder(StreamEncoder):
    """Preserve the pretrained Joint-TCN and learn a graph residual correction.

    Inheriting ``StreamEncoder`` intentionally keeps the original parameter names,
    so a TCN checkpoint can initialize this stream without key remapping.  The
    graph projection starts at zero: before fine-tuning the encoder is exactly the
    pretrained temporal expert, while gradients can gradually activate ST-GCN.
    """

    def __init__(
        self,
        channels: int,
        output_dim: int,
        dropout: float,
        *,
        recurrent_layers: int = 0,
        use_transformer_encoder: bool = False,
        use_gated_conv: bool = False,
    ) -> None:
        super().__init__(
            JOINT_DIM,
            channels,
            output_dim,
            dropout,
            recurrent_layers=recurrent_layers,
            use_transformer_encoder=use_transformer_encoder,
            use_gated_conv=use_gated_conv,
        )
        self.graph_residual = STGCNJointEncoder(channels, output_dim, dropout)
        nn.init.zeros_(self.graph_residual.output_projection.weight)
        nn.init.zeros_(self.graph_residual.output_projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) + self.graph_residual(x)


class MultiStreamMultiScaleAttentionTCN(nn.Module):
    """四流因果分类器，支持 Accel 或 Geometry+observed 第四流。"""

    def __init__(
        self,
        *,
        stream_channels: int = 60,
        stream_output_dim: int = 256,
        dropout: float = 0.2,
        use_geometry: bool = False,
        stream_lstm_layers: int = 0,
        use_rule_features: bool = False,
        use_pifr_features: bool = False,
        use_stgcn_joint: bool = False,
        use_stgcn_residual_joint: bool = False,
        use_transformer_encoder: bool = False,
        transformer_joint_only: bool = False,
        use_transition_rule_late_fusion: bool = False,
        use_discriminative_kinematics: bool = False,
        kinematic_feature_dim: int = KINEMATIC_RULE_DIM,
        use_gated_conv: bool = False,
        use_motion_guided_fusion: bool = False,
        motion_validity_mask: bool = False,
        per_joint_mask_correction: bool = False,
        use_stream_interaction: bool = False,
        use_stage_auxiliary: bool = False,
        use_physics_auxiliary: bool = False,
        use_conditional_verifier: bool = False,
        use_event_verifier: bool = False,
        event_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if stream_channels < 4 or stream_channels % 4:
            raise ValueError("stream_channels 必须是不小于 4 的 4 的倍数")
        if stream_output_dim < 1 or not 0 <= dropout < 1:
            raise ValueError("stream_output_dim 必须为正，dropout 必须位于 [0,1)")
        if stream_lstm_layers < 0:
            raise ValueError("stream_lstm_layers 不能为负")
        if event_hidden_dim < 1:
            raise ValueError("event_hidden_dim 必须为正")
        if use_conditional_verifier and use_event_verifier:
            raise ValueError("窗口 verifier 与事件 verifier 不能同时启用")
        if kinematic_feature_dim not in {12, KINEMATIC_RULE_DIM}:
            raise ValueError("运动学特征维度必须为 12 或当前版本维度")
        if (
            use_rule_features
            or use_pifr_features
            or use_transition_rule_late_fusion
            or use_discriminative_kinematics
        ) and not use_geometry:
            raise ValueError("规则/PIFR/转变/运动学特征需要 use_geometry")
        self.use_geometry = use_geometry
        self.stream_lstm_layers = stream_lstm_layers
        self.use_rule_features = use_rule_features
        self.use_pifr_features = use_pifr_features
        self.use_stgcn_joint = use_stgcn_joint
        self.use_stgcn_residual_joint = use_stgcn_residual_joint
        if use_stgcn_joint and use_stgcn_residual_joint:
            raise ValueError("ST-GCN replacement 与 residual 模式不能同时启用")
        self.use_transformer_encoder = use_transformer_encoder
        self.transformer_joint_only = transformer_joint_only
        if transformer_joint_only and not use_transformer_encoder:
            raise ValueError("transformer_joint_only 需要 use_transformer_encoder")
        self.use_transition_rule_late_fusion = use_transition_rule_late_fusion
        self.use_discriminative_kinematics = use_discriminative_kinematics
        self.kinematic_feature_dim = kinematic_feature_dim
        self.use_gated_conv = use_gated_conv
        self.use_motion_guided_fusion = use_motion_guided_fusion
        self.motion_validity_mask = motion_validity_mask
        self.per_joint_mask_correction = per_joint_mask_correction
        self.use_stream_interaction = use_stream_interaction
        self.use_stage_auxiliary = use_stage_auxiliary
        self.use_physics_auxiliary = use_physics_auxiliary
        self.use_conditional_verifier = use_conditional_verifier
        self.use_event_verifier = use_event_verifier
        stream_dimensions = (
            GEOMETRY_RULE_PIFR_STREAM_DIMENSIONS
            if use_rule_features and use_pifr_features
            else GEOMETRY_RULE_STREAM_DIMENSIONS
            if use_rule_features
            else GEOMETRY_PIFR_STREAM_DIMENSIONS
            if use_pifr_features
            else GEOMETRY_STREAM_DIMENSIONS
            if use_geometry
            else STREAM_DIMENSIONS
        )
        if use_discriminative_kinematics:
            stream_dimensions = (*stream_dimensions, kinematic_feature_dim)
        if motion_validity_mask:
            stream_dimensions = (
                stream_dimensions[0],
                stream_dimensions[1],
                MOTION_VALIDITY_DIM,
                *stream_dimensions[3:],
            )
        self.streams = nn.ModuleList(
            ResidualSTGCNJointEncoder(
                stream_channels,
                stream_output_dim,
                dropout,
                recurrent_layers=stream_lstm_layers,
                use_transformer_encoder=(
                    use_transformer_encoder
                    and (not transformer_joint_only or index == 0)
                ),
                use_gated_conv=use_gated_conv,
            )
            if use_stgcn_residual_joint and index == 0
            else STGCNJointEncoder(stream_channels, stream_output_dim, dropout)
            if use_stgcn_joint and index == 0
            else StreamEncoder(
                dimension,
                stream_channels,
                stream_output_dim,
                dropout,
                recurrent_layers=stream_lstm_layers,
                use_transformer_encoder=(
                    use_transformer_encoder
                    and (not transformer_joint_only or index == 0)
                ),
                use_gated_conv=use_gated_conv,
            )
            for index, dimension in enumerate(stream_dimensions)
        )
        fusion_dim = (
            stream_output_dim
            if use_motion_guided_fusion
            else stream_output_dim * len(stream_dimensions)
        )
        self.motion_fusion_attention = (
            nn.MultiheadAttention(
                stream_output_dim, num_heads=4, dropout=dropout, batch_first=True
            )
            if use_motion_guided_fusion
            else None
        )
        self.motion_fusion_norm = (
            nn.LayerNorm(stream_output_dim) if use_motion_guided_fusion else None
        )
        self.stream_interaction = (
            nn.MultiheadAttention(
                stream_output_dim, num_heads=2, dropout=dropout, batch_first=True
            )
            if use_stream_interaction
            else None
        )
        if self.stream_interaction is not None:
            nn.init.zeros_(self.stream_interaction.out_proj.weight)
            nn.init.zeros_(self.stream_interaction.out_proj.bias)
        reduced = max(8, fusion_dim // 16)
        self.fusion_se = nn.Sequential(
            nn.Linear(fusion_dim, reduced),
            nn.ReLU(),
            nn.Linear(reduced, fusion_dim),
            nn.Sigmoid(),
        )
        self.classifier = nn.Linear(fusion_dim, 2)
        self.event_projection = (
            nn.Sequential(
                nn.Linear(fusion_dim, event_hidden_dim),
                nn.LayerNorm(event_hidden_dim),
                nn.GELU(),
            )
            if use_event_verifier
            else None
        )
        self.event_encoder = (
            nn.GRU(event_hidden_dim, event_hidden_dim, batch_first=True)
            if use_event_verifier
            else None
        )
        self.event_classifier = (
            nn.Linear(event_hidden_dim, 1) if use_event_verifier else None
        )
        if self.event_classifier is not None:
            nn.init.zeros_(self.event_classifier.weight)
            nn.init.zeros_(self.event_classifier.bias)
            self.register_buffer("event_verifier_scale", torch.tensor(0.0))
        else:
            self.event_verifier_scale = None
        self.stage_classifier = (
            nn.Linear(fusion_dim, 4) if use_stage_auxiliary else None
        )
        self.verifier_classifier = (
            nn.Linear(fusion_dim, 4) if use_conditional_verifier else None
        )
        if self.verifier_classifier is not None:
            # A zero verifier is an exact no-op. This preserves the initialized
            # candidate checkpoint and lets the specialist learn a residual
            # fall-vs-controlled-transition decision.
            nn.init.zeros_(self.verifier_classifier.weight)
            nn.init.zeros_(self.verifier_classifier.bias)
            self.register_buffer("verifier_scale", torch.tensor(0.0))
        else:
            self.verifier_scale = None
        self.physics_auxiliary = nn.Linear(fusion_dim, 3) if use_physics_auxiliary else None
        self.transition_stream = (
            StreamEncoder(
                TRANSITION_RULE_DIM, stream_channels, stream_output_dim, dropout
            )
            if use_transition_rule_late_fusion
            else None
        )
        self.transition_classifier = (
            nn.Linear(stream_output_dim, 1) if use_transition_rule_late_fusion else None
        )
        self.transition_gate = (
            nn.Linear(fusion_dim + stream_output_dim, 1)
            if use_transition_rule_late_fusion
            else None
        )
        if self.transition_classifier is not None and self.transition_gate is not None:
            nn.init.zeros_(self.transition_classifier.weight)
            nn.init.zeros_(self.transition_classifier.bias)
            nn.init.zeros_(self.transition_gate.weight)
            nn.init.constant_(self.transition_gate.bias, -2.0)

    def _stream_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        streams = build_stream_features(
            x,
            use_geometry=self.use_geometry,
            use_rules=self.use_rule_features,
            use_pifr=self.use_pifr_features,
            use_kinematics=self.use_discriminative_kinematics,
            kinematic_feature_dim=self.kinematic_feature_dim,
            motion_validity_mask=self.motion_validity_mask,
            per_joint_mask_correction=self.per_joint_mask_correction,
        )
        embeddings = torch.stack(
            [
                encoder(feature)
                for encoder, feature in zip(self.streams, streams, strict=True)
            ],
            dim=1,
        )
        if self.stream_interaction is not None:
            interacted, _ = self.stream_interaction(
                embeddings, embeddings, embeddings, need_weights=False
            )
            embeddings = embeddings + interacted
        return embeddings

    def _encode_fused(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the main multi-stream evidence before final classification."""
        embeddings = self._stream_embeddings(x)
        if (
            self.motion_fusion_attention is not None
            and self.motion_fusion_norm is not None
        ):
            motion_query = embeddings[:, 2:3]
            attended, _ = self.motion_fusion_attention(
                motion_query, embeddings, embeddings, need_weights=False
            )
            fused = self.motion_fusion_norm(motion_query + attended).squeeze(1)
        else:
            fused = embeddings.flatten(1)
        return fused * self.fusion_se(fused)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the fused representation for audited downstream ensembles."""
        return self._encode_fused(x)

    def motion_fusion_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-sample attention over streams for the motion-guided model."""
        if self.motion_fusion_attention is None:
            raise ValueError("模型未启用 motion-guided fusion")
        embeddings = self._stream_embeddings(x)
        _, weights = self.motion_fusion_attention(
            embeddings[:, 2:3], embeddings, embeddings, need_weights=True
        )
        return weights.squeeze(1)

    def forward_class_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._encode_fused(x))

    def forward_with_transition(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fused fall logit and independently supervised transition logit."""
        if (
            self.transition_stream is None
            or self.transition_classifier is None
            or self.transition_gate is None
        ):
            raise ValueError("模型未启用转变规则 late fusion")
        fused = self._encode_fused(x)
        main_logits = self.classifier(fused)
        main_logit = main_logits[:, 1] - main_logits[:, 0]
        transition_embedding = self.transition_stream(build_transition_rule_features(x))
        transition_logit = self.transition_classifier(transition_embedding).squeeze(-1)
        gate = torch.sigmoid(
            self.transition_gate(torch.cat((fused, transition_embedding), dim=-1))
        ).squeeze(-1)
        return main_logit + gate * transition_logit, transition_logit

    def forward_with_stage(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return binary fall logit and four-way temporal-stage logits."""
        if self.stage_classifier is None:
            raise ValueError("模型未启用阶段辅助监督")
        fused = self._encode_fused(x)
        main_logits = self.classifier(fused)
        return main_logits[:, 1] - main_logits[:, 0], self.stage_classifier(fused)

    def forward_with_verifier(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return final, candidate, and four-way conditional-verifier logits.

        Verifier classes are uncontrolled fall, controlled descent, reverse
        transition, and stationary/background. The centered log-mean-exp
        contrast is zero for a zero-initialized head, so loading an existing
        candidate checkpoint preserves its predictions exactly.
        """
        if self.verifier_classifier is None or self.verifier_scale is None:
            raise ValueError("模型未启用受控转变验证器")
        fused = self._encode_fused(x)
        candidate_logits = self.classifier(fused)
        candidate_logit = candidate_logits[:, 1] - candidate_logits[:, 0]
        verifier_logits = self.verifier_classifier(fused)
        fall_evidence = torch.logsumexp(verifier_logits[:, :1], dim=1)
        nonfall_evidence = torch.logsumexp(verifier_logits[:, 1:], dim=1)
        verifier_delta = fall_evidence - nonfall_evidence + math.log(3.0)
        return (
            candidate_logit + self.verifier_scale * verifier_delta,
            candidate_logit,
            verifier_logits,
        )

    def set_verifier_scale(self, value: float) -> None:
        """Set the audited residual-verifier contribution for training/eval."""
        if self.verifier_scale is None:
            raise ValueError("模型未启用受控转变验证器")
        if not 0.0 <= value <= 1.0:
            raise ValueError("verifier scale 必须位于 [0, 1]")
        self.verifier_scale.fill_(value)

    def event_verifier_delta(
        self, fused: torch.Tensor, group_sizes: list[int]
    ) -> torch.Tensor:
        """Encode ordered candidate-window embeddings into one delta per clip."""
        if (
            self.event_projection is None
            or self.event_encoder is None
            or self.event_classifier is None
        ):
            raise ValueError("模型未启用事件级验证器")
        if fused.ndim != 2 or not group_sizes or sum(group_sizes) != fused.shape[0]:
            raise ValueError("fused 与 group_sizes 不匹配")
        sequences = list(torch.split(self.event_projection(fused), group_sizes, dim=0))
        padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        lengths = torch.tensor(group_sizes, device="cpu")
        packed = nn.utils.rnn.pack_padded_sequence(
            padded, lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.event_encoder(packed)
        return self.event_classifier(hidden[-1]).squeeze(-1)

    def set_event_verifier_scale(self, value: float) -> None:
        if self.event_verifier_scale is None:
            raise ValueError("模型未启用事件级验证器")
        if not 0.0 <= value <= 1.0:
            raise ValueError("event verifier scale 必须位于 [0, 1]")
        self.event_verifier_scale.fill_(value)

    def forward_with_physics(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.physics_auxiliary is None:
            raise ValueError("模型未启用物理辅助监督")
        fused = self._encode_fused(x)
        logits = self.classifier(fused)
        return logits[:, 1] - logits[:, 0], self.physics_auxiliary(fused)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """兼容 BCE 训练器：返回 positive-minus-negative 二分类 logit。"""
        if self.use_transition_rule_late_fusion:
            return self.forward_with_transition(x)[0]
        if self.use_conditional_verifier:
            return self.forward_with_verifier(x)[0]
        logits = self.forward_class_logits(x)
        return logits[:, 1] - logits[:, 0]


def count_params(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
