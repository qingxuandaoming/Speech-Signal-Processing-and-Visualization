#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预处理（Preprocessing）

提供语音信号的预加重与分帧加窗等常用预处理函数。
"""

import numpy as np

from .windows import hamming_window, hanning_window, rectangular_window


def preemphasis(signal: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    """
    预加重（Pre-emphasis）。

    对时域信号进行一阶高通预加重，提升高频成分，有助于后续特征提取。

    Parameters
    ----------
    signal : numpy.ndarray
        输入时域信号，一维数组。
    alpha : float, default=0.97
        预加重系数，取值范围通常在 [0.9, 0.98]。

    Returns
    -------
    numpy.ndarray
        预加重后的时域信号，与输入长度相同。
    """
    if signal.size == 0:
        return signal.astype(np.float32)
    signal = signal.astype(np.float32, copy=False)
    return np.append(signal[0], signal[1:] - alpha * signal[:-1]).astype(np.float32)


def framing(
    signal: np.ndarray,
    frame_size: int,
    hop_size: int,
    window_type: str = "hamming",
) -> np.ndarray:
    """
    分帧并加窗（Framing & Windowing）。

    将连续时域信号按给定帧长与帧移分割为重叠帧，并应用指定窗函数。

    Parameters
    ----------
    signal : numpy.ndarray
        输入时域信号，一维数组。
    frame_size : int
        帧长度（样本点）。
    hop_size : int
        帧移（样本点）。
    window_type : {"hamming", "hanning", "rectangular"}, default="hamming"
        加窗类型。

    Returns
    -------
    numpy.ndarray
        形状为 `(num_frames, frame_size)` 的二维数组，每行对应一帧。

    Notes
    -----
    若信号长度不足以整帧分割，则在末尾进行零填充以满足索引需求。
    """
    signal = signal.astype(np.float32, copy=False)
    signal_length = int(signal.size)
    if frame_size <= 0 or hop_size <= 0 or signal_length == 0:
        return np.zeros((0, max(frame_size, 0)), dtype=np.float32)

    num_frames = 1 + int(np.ceil((signal_length - frame_size) / hop_size))
    pad_length = (num_frames - 1) * hop_size + frame_size
    padded_signal = np.pad(signal, (0, max(0, pad_length - signal_length)), mode="constant")

    indices = (
        np.tile(np.arange(0, frame_size), (num_frames, 1))
        + np.tile(np.arange(0, num_frames * hop_size, hop_size), (frame_size, 1)).T
    ).astype(np.int32, copy=False)

    frames = padded_signal[indices]

    if window_type == "hamming":
        window = hamming_window(frame_size)
    elif window_type == "hanning":
        window = hanning_window(frame_size)
    else:
        window = rectangular_window(frame_size)

    return (frames * window).astype(np.float32)