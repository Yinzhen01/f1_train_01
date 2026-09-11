"""Yaw-independent tilt tracking for a fixed-waist reference (CPU/CUDA tensors)."""
import torch


def tilt_error_squared(actual_gravity, target_gravity):
    actual = actual_gravity / torch.linalg.vector_norm(actual_gravity, dim=-1, keepdim=True).clamp_min(1e-8)
    target = target_gravity / torch.linalg.vector_norm(target_gravity, dim=-1, keepdim=True).clamp_min(1e-8)
    return (actual - target).square().sum(dim=-1)


def tilt_tracking_reward(actual_gravity, target_gravity, sigma):
    if sigma <= 0:
        raise ValueError('Tilt sigma must be positive')
    return torch.exp(-tilt_error_squared(actual_gravity, target_gravity) / sigma ** 2)
