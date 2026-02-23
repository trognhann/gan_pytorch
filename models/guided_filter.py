import torch
import torch.nn as nn


def diff_x(input, r):
    left = input[:, :, :, r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:] - input[:, :, :, :-2 * r - 1]
    right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1:-r - 1]
    return torch.cat([left, middle, right], dim=3)


def diff_y(input, r):
    left = input[:, :, r:2 * r + 1, :]
    middle = input[:, :, 2 * r + 1:, :] - input[:, :, :-2 * r - 1, :]
    right = input[:, :, -1:, :] - input[:, :, -2 * r - 1:-r - 1, :]
    return torch.cat([left, middle, right], dim=2)


def box_filter(x, r):
    return diff_y(torch.cumsum(diff_x(torch.cumsum(x, dim=3), r), dim=2), r)


def guided_filter(x, y, r, eps=1e-1):
    """
    x: guidance image with shape [B, C, H, W]
    y: filtering input image with shape [B, C, H, W]
    """
    N = box_filter(torch.ones_like(x), r)

    mean_x = box_filter(x, r) / N
    mean_y = box_filter(y, r) / N
    cov_xy = box_filter(x * y, r) / N - mean_x * mean_y
    var_x = box_filter(x * x, r) / N - mean_x * mean_x

    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x

    mean_A = box_filter(A, r) / N
    mean_b = box_filter(b, r) / N

    output = mean_A * x + mean_b
    return output
