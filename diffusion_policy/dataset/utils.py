import torch
from .math_utils import quat_mul, quat_inv, axis_angle_from_quat, quat_from_angle_axis

SCALE = torch.tensor([0.02, 0.02, 0.02, 0.02, 0.02, 0.2])

def pose_axis_angle_to_pos_quat(pose_aa: torch.Tensor):
    pos = pose_aa[:, :3]
    rot_aa = pose_aa[:, 3:6]
    angle = torch.norm(rot_aa, dim=-1)
    safe_angle = torch.where(angle > 1e-6, angle, torch.ones_like(angle))
    axis = rot_aa / safe_angle.unsqueeze(-1)
    axis = torch.where(angle.unsqueeze(-1) > 1e-6, axis, torch.zeros_like(axis))
    quat = quat_from_angle_axis(angle, axis)
    return pos, quat

def process_actions(
    actions: torch.Tensor,
    ee_pos_ref: torch.Tensor,
    ee_quat_ref: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map raw 6-DOF actions to a desired EE pose relative to a reference pose.
    Args:
        actions: Raw action tensor of shape (N, 6).
        ee_pos_ref: Reference EE position in base/root frame, shape (N, 3).
        ee_quat_ref: Reference EE orientation in base/root frame, shape (N, 4), (w, x, y, z).
    Returns:
        Tuple of:
            - ee_pos_des: Desired EE position, shape (N, 3)
            - ee_quat_des: Desired EE orientation, shape (N, 4)
    """
    SCALE.to(actions.device)
    scaled = actions * SCALE
    # Position delta in base frame
    ee_pos_des = ee_pos_ref + scaled[:, :3]
    # Axis-angle delta -> quaternion
    delta_rot = scaled[:, 3:6]
    angle = torch.norm(delta_rot, dim=-1, keepdim=True)
    safe_angle = torch.where(angle > 1e-6, angle, torch.ones_like(angle))
    axis = delta_rot / safe_angle
    axis = torch.where(angle > 1e-6, axis, torch.zeros_like(axis))
    half = angle / 2.0
    delta_quat = torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1)
    # Apply delta relative to explicit reference orientation
    ee_quat_des = quat_mul(delta_quat, ee_quat_ref)
    return ee_pos_des, ee_quat_des

def inverse_process_actions(
    ee_pos_des: torch.Tensor,
    ee_quat_des: torch.Tensor,
    ee_pos_ref: torch.Tensor,
    ee_quat_ref: torch.Tensor,
) -> torch.Tensor:
    """Recover raw 6-DOF actions from a desired EE pose and reference pose.
    Args:
        ee_pos_des: Desired EE position in base/root frame, shape (N, 3).
        ee_quat_des: Desired EE orientation in base/root frame, shape (N, 4).
        ee_pos_ref: Reference EE position in base/root frame, shape (N, 3).
        ee_quat_ref: Reference EE orientation in base/root frame, shape (N, 4).
    Returns:
        Raw action tensor of shape (N, 6).
    """
    SCALE.to(ee_pos_des.device)
    # Position inverse
    delta_pos = ee_pos_des - ee_pos_ref
    # Orientation inverse:
    # ee_quat_des = delta_quat * ee_quat_ref
    # => delta_quat = ee_quat_des * inv(ee_quat_ref)
    delta_quat = quat_mul(ee_quat_des, quat_inv(ee_quat_ref))
    delta_rot = axis_angle_from_quat(delta_quat)
    scaled = torch.cat([delta_pos, delta_rot], dim=-1)
    raw_actions = scaled / SCALE
    return raw_actions