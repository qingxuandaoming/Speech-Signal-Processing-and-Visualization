#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信号处理包（signal_processing）

提供语音信号处理的常用算法与工具的模块化实现，并通过
`SignalProcessing` 聚合类保留与原版本兼容的静态方法接口。

该包拆分为以下子模块：

- `windows`：常用窗函数生成
- `preprocessing`：预加重、分帧与加窗
- `time_features`：时域特征（短时能量、过零率、自相关、AMDF）
- `frequency_features`：频域特征（Mel滤波器、MFCC、谱熵）
- `vad`：语音活动检测（固定阈值与自适应）

Notes
-----
为兼容历史代码，`SignalProcessing` 静态方法名称与签名保持不变，
具体实现委托给各子模块的函数。
"""

from .windows import hamming_window as _hamming_window, hanning_window as _hanning_window, rectangular_window as _rectangular_window
from .preprocessing import preemphasis as _preemphasis, framing as _framing
from .time_features import (
    calculate_short_time_energy as _calculate_short_time_energy,
    calculate_zero_crossing_rate as _calculate_zero_crossing_rate,
    calculate_short_time_autocorrelation as _calculate_short_time_autocorrelation,
    calculate_average_magnitude_difference as _calculate_average_magnitude_difference,
)
from .frequency_features import (
    mel_filterbank as _mel_filterbank,
    compute_mfcc as _compute_mfcc,
    calculate_spectral_entropy as _calculate_spectral_entropy,
)
from .vad import voice_activity_detection as _voice_activity_detection, adaptive_voice_activity_detection as _adaptive_voice_activity_detection
import numpy as np
try:
    from real_time_voice_processing.config import Config as _Config
except Exception:  # 允许在独立测试子包时缺失顶层包
    _Config = None


class SignalProcessing:
    """
    信号处理算法聚合类。

    该类提供与历史版本一致的静态方法接口，内部委托到
    `signal_processing` 包各子模块的函数实现。

    See Also
    --------
    windows : 窗函数生成函数
    preprocessing : 预加重与分帧函数
    time_features : 时域特征计算函数
    frequency_features : 频域特征计算函数
    vad : 语音活动检测函数
    """

    # 窗函数
    @staticmethod
    def hamming_window(length: int):
        """包装 `windows.hamming_window` 保持签名与行为不变。"""
        return _hamming_window(length)

    @staticmethod
    def hanning_window(length: int):
        """包装 `windows.hanning_window` 保持签名与行为不变。"""
        return _hanning_window(length)

    @staticmethod
    def rectangular_window(length: int):
        """包装 `windows.rectangular_window` 保持签名与行为不变。"""
        return _rectangular_window(length)

    # 预处理
    @staticmethod
    def preemphasis(signal: np.ndarray, alpha: float = 0.97):
        """包装 `preprocessing.preemphasis`。"""
        return _preemphasis(signal, alpha=alpha)

    @staticmethod
    def framing(signal: np.ndarray, frame_size: int, hop_size: int, window_type: str = "hamming"):
        """包装 `preprocessing.framing`。"""
        return _framing(signal, frame_size=frame_size, hop_size=hop_size, window_type=window_type)

    # 时域特征
    @staticmethod
    def calculate_short_time_energy(frames_or_frame: np.ndarray):
        """
        兼容计算短时能量：支持传入单帧（一维）或多帧（二维）。

        如果输入是一维数组，返回单个标量能量；若为二维数组，返回每帧能量数组。
        """
        arr = np.asarray(frames_or_frame, dtype=np.float32)
        if arr.ndim == 1:
            return float(np.sum(arr ** 2))
        return _calculate_short_time_energy(arr)

    @staticmethod
    def calculate_zero_crossing_rate(frames_or_frame: np.ndarray):
        """
        兼容计算过零率：支持单帧（一维）或多帧（二维）。

        一维输入返回标量 ZCR，二维输入返回每帧 ZCR 数组。
        """
        arr = np.asarray(frames_or_frame, dtype=np.float32)
        if arr.ndim == 1:
            signs = np.sign(arr)
            crossings = np.sum(np.abs(np.diff(signs)) > 0)
            return float(crossings) / arr.size if arr.size else 0.0
        return _calculate_zero_crossing_rate(arr)

    @staticmethod
    def calculate_short_time_autocorrelation(frames: np.ndarray, max_lag: int):
        """
        兼容包装：单帧输入返回一维归一化 ACF，长度为 `max_lag`，使得 `acf[0] == 1.0`。
        多帧输入返回二维数组。
        """
        frames = np.atleast_2d(frames).astype(np.float32)
        acf = _calculate_short_time_autocorrelation(frames, max_lag=max_lag)
        if np.asarray(frames).ndim == 2 and frames.shape[0] == 1:
            vec = acf[0, :max_lag].astype(np.float32)
            if vec.size and vec[0] != 0:
                vec = (vec / vec[0]).astype(np.float32)
            return vec
        return acf

    @staticmethod
    def calculate_average_magnitude_difference(frames: np.ndarray, max_lag: int):
        """包装 AMDF 计算。"""
        frames = np.atleast_2d(frames).astype(np.float32)
        return _calculate_average_magnitude_difference(frames, max_lag=max_lag)

    # 频域特征
    @staticmethod
    def mel_filterbank(n_filters: int, n_fft: int, sample_rate: int, fmin: float = 0.0, fmax: float | None = None):
        """
        兼容包装：参数名 `n_filters` 映射至实现函数的 `num_filters`。
        """
        return _mel_filterbank(num_filters=n_filters, n_fft=n_fft, sample_rate=sample_rate, fmin=fmin, fmax=fmax)

    @staticmethod
    def compute_mfcc(
        frame_or_frames: np.ndarray,
        sample_rate: int,
        n_fft: int = 512,
        n_filters: int = 26,
        num_ceps: int = 13,
        lifter: int | None = None,
        pre_emphasis: float | None = None,
        fmin: float = 0.0,
        fmax: float | None = None,
    ):
        """
        兼容包装：支持一维单帧输入与历史参数命名（`n_filters`、`lifter`、`pre_emphasis`）。
        """
        frames = np.atleast_2d(frame_or_frames).astype(np.float32)
        if pre_emphasis is not None and pre_emphasis > 0:
            # 可选预加重
            frames = np.vstack([_preemphasis(fr, alpha=pre_emphasis) for fr in frames])
        mfcc = _compute_mfcc(
            frames,
            sample_rate=sample_rate,
            n_fft=n_fft,
            num_filters=n_filters,
            num_ceps=num_ceps,
            fmin=fmin,
            fmax=fmax,
        )
        if lifter is not None and lifter > 0:
            n = np.arange(num_ceps)
            lift = 1.0 + (lifter / 2.0) * np.sin(np.pi * n / lifter)
            mfcc = mfcc * lift
        # 若输入为单帧则返回一维数组
        return mfcc[0] if np.asarray(frame_or_frames).ndim == 1 else mfcc

    @staticmethod
    def calculate_spectral_entropy(frame_or_frames: np.ndarray, n_fft: int = 512):
        """
        兼容包装：支持单帧（一维）或多帧（二维）。
        """
        frames = np.atleast_2d(frame_or_frames).astype(np.float32)
        entropy = _calculate_spectral_entropy(frames, n_fft=n_fft)
        return float(entropy[0]) if np.asarray(frame_or_frames).ndim == 1 else entropy

    # 语音活动检测
    @staticmethod
    def voice_activity_detection(
        energy,
        zcr,
        energy_threshold: float | None = None,
        zcr_threshold: float | None = None,
    ):
        """
        兼容固定阈值 VAD：支持标量或数组输入；阈值缺省时从 `Config` 读取。
        返回与输入维度相匹配的布尔/整型结果（单值或数组）。
        """
        if energy_threshold is None and _Config is not None:
            energy_threshold = _Config.ENERGY_THRESHOLD
        if zcr_threshold is None and _Config is not None:
            zcr_threshold = _Config.ZCR_THRESHOLD

        energy_arr = np.atleast_1d(np.asarray(energy, dtype=np.float32))
        zcr_arr = np.atleast_1d(np.asarray(zcr, dtype=np.float32))
        result = _voice_activity_detection(energy_arr, zcr_arr, float(energy_threshold or 0.0), float(zcr_threshold or 0.0))
        if np.asarray(energy).ndim == 0 and np.asarray(zcr).ndim == 0:
            return int(bool(result[0]))
        return result

    @staticmethod
    def adaptive_voice_activity_detection(
        energy,
        zcr,
        energy_history: list[float],
        zcr_history: list[float],
        **kwargs,
    ):
        """
        兼容自适应 VAD：接受历史版本参数名并忽略未使用的项（例如 `energy_k`、`zcr_k`、`min_history`、`fallback_*`）。
        支持标量或数组输入，返回相应布尔结果。
        """
        # 映射可选 alpha
        alpha = kwargs.get("alpha")
        if alpha is None:
            # 若提供 energy_k / zcr_k 则优先使用其中之一
            for k in ("energy_k", "zcr_k"):
                if k in kwargs and kwargs[k] is not None:
                    try:
                        alpha = float(kwargs[k])
                    except Exception:
                        alpha = 0.8
                    break
        if alpha is None:
            alpha = 0.8

        min_energy_threshold = float(kwargs.get("min_energy_threshold", 1e-6))
        max_zcr_threshold = float(kwargs.get("max_zcr_threshold", 0.5))

        energy_arr = np.atleast_1d(np.asarray(energy, dtype=np.float32))
        zcr_arr = np.atleast_1d(np.asarray(zcr, dtype=np.float32))
        result = _adaptive_voice_activity_detection(
            energy_arr,
            zcr_arr,
            list(energy_history) if energy_history is not None else [],
            list(zcr_history) if zcr_history is not None else [],
            alpha=alpha,
            min_energy_threshold=min_energy_threshold,
            max_zcr_threshold=max_zcr_threshold,
        )
        if np.asarray(energy).ndim == 0 and np.asarray(zcr).ndim == 0:
            return bool(result[0])
        return result


__all__ = [
    "SignalProcessing",
    # 函数也允许直接导入使用
    "hamming_window",
    "hanning_window",
    "rectangular_window",
    "preemphasis",
    "framing",
    "calculate_short_time_energy",
    "calculate_zero_crossing_rate",
    "calculate_short_time_autocorrelation",
    "calculate_average_magnitude_difference",
    "mel_filterbank",
    "compute_mfcc",
    "calculate_spectral_entropy",
    "voice_activity_detection",
    "adaptive_voice_activity_detection",
]