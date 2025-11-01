#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
频域特征（Frequency-domain Features）

提供 Mel 滤波器组、MFCC 与谱熵计算。
"""

import numpy as np
from scipy.fftpack import dct


def _hz_to_mel(freq_hz: np.ndarray) -> np.ndarray:
    """
    Hz 转 Mel 标度。

    Parameters
    ----------
    freq_hz : numpy.ndarray
        以 Hz 为单位的频率数组。

    Returns
    -------
    numpy.ndarray
        对应的 Mel 标度数值。
    """
    return 2595 * np.log10(1 + freq_hz / 700.0)


def _mel_to_hz(freq_mel: np.ndarray) -> np.ndarray:
    """
    Mel 标度转 Hz。

    Parameters
    ----------
    freq_mel : numpy.ndarray
        以 Mel 为单位的频率数组。

    Returns
    -------
    numpy.ndarray
        对应的 Hz 频率数值。
    """
    return 700 * (10 ** (freq_mel / 2595.0) - 1)


def mel_filterbank(
    num_filters: int,
    n_fft: int,
    sample_rate: int,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """
    构造 Mel 滤波器组。

    Parameters
    ----------
    num_filters : int
        滤波器数量（通常在 20~40）。
    n_fft : int
        FFT 点数（谱线数）。
    sample_rate : int
        采样率（Hz）。
    fmin : float, default=0.0
        最低频率（Hz）。
    fmax : float or None, default=None
        最高频率（Hz）；若为 None 则取 `sample_rate/2`。

    Returns
    -------
    numpy.ndarray
        形状为 `(num_filters, n_fft//2 + 1)` 的滤波器系数矩阵。
    """
    if fmax is None:
        fmax = sample_rate / 2

    # 等间距 Mel 刻度点
    mel_min = _hz_to_mel(np.array([fmin]))[0]
    mel_max = _hz_to_mel(np.array([fmax]))[0]
    mel_points = np.linspace(mel_min, mel_max, num_filters + 2)
    hz_points = _mel_to_hz(mel_points)

    # 频率对应的谱线索引
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filterbank = np.zeros((num_filters, n_fft // 2 + 1), dtype=np.float32)

    for i in range(1, num_filters + 1):
        left, center, right = bin_points[i - 1], bin_points[i], bin_points[i + 1]
        if center == left:
            center += 1
        if right == center:
            right += 1
        # 上升段
        filterbank[i - 1, left:center] = (
            (np.arange(left, center) - left) / (center - left)
        )
        # 下降段
        filterbank[i - 1, center:right] = (
            (right - np.arange(center, right)) / (right - center)
        )

    # 只保留正频谱
    return filterbank[:, : (n_fft // 2 + 1)].astype(np.float32)


def compute_mfcc(
    frames: np.ndarray,
    sample_rate: int,
    n_fft: int = 512,
    num_filters: int = 26,
    num_ceps: int = 13,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """
    计算梅尔频率倒谱系数（MFCC）。

    Parameters
    ----------
    frames : numpy.ndarray
        形状为 `(num_frames, frame_size)` 的加窗帧数组。
    sample_rate : int
        采样率（Hz）。
    n_fft : int, default=512
        FFT 点数。
    num_filters : int, default=26
        Mel 滤波器数量。
    num_ceps : int, default=13
        倒谱系数数量。
    fmin : float, default=0.0
        最低频率（Hz）。
    fmax : float or None, default=None
        最高频率（Hz）；若为 None 则取 `sample_rate/2`。

    Returns
    -------
    numpy.ndarray
        形状为 `(num_frames, num_ceps)` 的 MFCC 特征矩阵。
    """
    frames = frames.astype(np.float32, copy=False)
    if frames.size == 0:
        return np.zeros((0, num_ceps), dtype=np.float32)

    # FFT 与功率谱
    spectrum = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2

    # Mel 滤波器组
    fb = mel_filterbank(num_filters=num_filters, n_fft=n_fft, sample_rate=sample_rate, fmin=fmin, fmax=fmax)

    # 滤波与对数变换
    energy = np.maximum(np.dot(spectrum, fb.T), 1e-10)
    log_energy = np.log(energy)

    # DCT 获得 MFCC
    mfcc = dct(log_energy, type=2, axis=1, norm="ortho")[:, :num_ceps]
    return mfcc.astype(np.float32)


def calculate_spectral_entropy(frames: np.ndarray, n_fft: int = 512) -> np.ndarray:
    """
    计算谱熵（Spectral Entropy）。

    谱熵衡量频谱分布的离散程度，越高表示越均匀。

    Parameters
    ----------
    frames : numpy.ndarray
        形状为 `(num_frames, frame_size)` 的加窗帧数组。
    n_fft : int, default=512
        FFT 点数。

    Returns
    -------
    numpy.ndarray
        每帧谱熵，一维数组长度为 `num_frames`。
    """
    frames = frames.astype(np.float32, copy=False)
    if frames.size == 0:
        return np.array([], dtype=np.float32)

    spectrum = np.abs(np.fft.rfft(frames, n=n_fft))
    psd = spectrum ** 2
    psd_sum = np.sum(psd, axis=1, keepdims=True)
    psd_norm = np.divide(psd, psd_sum, where=psd_sum > 0)

    # 避免 log(0)
    psd_norm = np.maximum(psd_norm, 1e-12)
    entropy = -np.sum(psd_norm * np.log(psd_norm), axis=1)

    # 归一化到 [0, 1]
    num_bins = psd.shape[1]
    max_entropy = np.log(num_bins)
    entropy_norm = np.divide(entropy, max_entropy, where=max_entropy > 0)
    return entropy_norm.astype(np.float32)