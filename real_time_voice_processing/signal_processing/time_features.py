#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时域特征（Time-domain Features）

提供短时能量、过零率（ZCR）、短时自相关与平均幅度差（AMDF）等特征计算。
"""

import numpy as np


def calculate_short_time_energy(frames: np.ndarray) -> np.ndarray:
    """
    计算短时能量（Short-time Energy）。

    Parameters
    ----------
    frames : numpy.ndarray
        形状为 `(num_frames, frame_size)` 的加窗帧数组。

    Returns
    -------
    numpy.ndarray
        每帧能量，一维数组长度为 `num_frames`。
    """
    if frames.size == 0:
        return np.array([], dtype=np.float32)
    return np.sum(frames.astype(np.float32) ** 2, axis=1).astype(np.float32)


def calculate_zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    """
    计算过零率（Zero-crossing Rate, ZCR）。

    Parameters
    ----------
    frames : numpy.ndarray
        形状为 `(num_frames, frame_size)` 的加窗帧数组。

    Returns
    -------
    numpy.ndarray
        每帧的 ZCR，一维数组长度为 `num_frames`。
    """
    if frames.size == 0:
        return np.array([], dtype=np.float32)
    signs = np.sign(frames)
    crossings = np.sum(np.abs(np.diff(signs, axis=1)) > 0, axis=1)
    return crossings.astype(np.float32) / frames.shape[1]


def calculate_short_time_autocorrelation(frames: np.ndarray, max_lag: int) -> np.ndarray:
    """
    计算短时自相关（Short-time Autocorrelation）。

    Parameters
    ----------
    frames : numpy.ndarray
        形状为 `(num_frames, frame_size)` 的加窗帧数组。
    max_lag : int
        最大滞后量（样本点）。

    Returns
    -------
    numpy.ndarray
        形状为 `(num_frames, max_lag+1)` 的二维数组，每行对应不同滞后量的自相关值。
    """
    frames = frames.astype(np.float32, copy=False)
    num_frames, frame_size = frames.shape if frames.size else (0, 0)
    if num_frames == 0 or max_lag < 0:
        return np.zeros((num_frames, max(0, max_lag + 1)), dtype=np.float32)

    result = np.zeros((num_frames, max_lag + 1), dtype=np.float32)
    for lag in range(max_lag + 1):
        result[:, lag] = np.sum(frames[:, :-lag or None] * frames[:, lag:], axis=1)
    return result


def calculate_average_magnitude_difference(frames: np.ndarray, max_lag: int) -> np.ndarray:
    """
    计算平均幅度差函数（Average Magnitude Difference Function, AMDF）。

    Parameters
    ----------
    frames : numpy.ndarray
        形状为 `(num_frames, frame_size)` 的加窗帧数组。
    max_lag : int
        最大滞后量（样本点）。

    Returns
    -------
    numpy.ndarray
        形状为 `(num_frames, max_lag)` 的二维数组，每行对应不同滞后量的 AMDF 值。
    """
    frames = frames.astype(np.float32, copy=False)
    num_frames, frame_size = frames.shape if frames.size else (0, 0)
    if num_frames == 0 or max_lag <= 0:
        return np.zeros((num_frames, max(0, max_lag)), dtype=np.float32)

    result = np.zeros((num_frames, max_lag), dtype=np.float32)
    for lag in range(1, max_lag + 1):
        diff = np.abs(frames[:, :-lag] - frames[:, lag:])
        result[:, lag - 1] = np.mean(diff, axis=1)
    return result