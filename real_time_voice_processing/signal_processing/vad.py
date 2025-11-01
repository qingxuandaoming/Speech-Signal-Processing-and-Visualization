#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音活动检测（Voice Activity Detection, VAD）

提供基于短时能量与过零率的固定阈值 VAD，以及使用历史统计的自适应 VAD。
"""

import numpy as np


def voice_activity_detection(
    energy: np.ndarray,
    zcr: np.ndarray,
    energy_threshold: float,
    zcr_threshold: float,
) -> np.ndarray:
    """
    基于能量与过零率的固定阈值 VAD。

    Parameters
    ----------
    energy : numpy.ndarray
        每帧短时能量数组，形状为 `(num_frames,)` 或标量。
    zcr : numpy.ndarray
        每帧过零率数组，形状为 `(num_frames,)` 或标量。
    energy_threshold : float
        能量阈值，高于该值视为语音。
    zcr_threshold : float
        过零率阈值，高于该值更可能为语音。

    Returns
    -------
    numpy.ndarray
        形状为 `(num_frames,)` 的布尔数组，`True` 表示语音活动。
    """
    energy = energy.astype(np.float32, copy=False)
    zcr = zcr.astype(np.float32, copy=False)
    # 历史项目的判定：能量高且 ZCR 高判为语音
    vad = (energy > energy_threshold) & (zcr > zcr_threshold)
    return vad.astype(bool)


def adaptive_voice_activity_detection(
    energy: np.ndarray,
    zcr: np.ndarray,
    energy_history: list[float],
    zcr_history: list[float],
    alpha: float = 0.8,
    min_energy_threshold: float = 1e-6,
    max_zcr_threshold: float = 0.5,
) -> np.ndarray:
    """
    自适应 VAD（Adaptive VAD）。

    根据能量与过零率的历史统计自适应调整阈值，提升稳健性。

    Parameters
    ----------
    energy : numpy.ndarray
        当前批次每帧能量数组，形状为 `(num_frames,)`。
    zcr : numpy.ndarray
        当前批次每帧过零率数组，形状为 `(num_frames,)`。
    energy_history : list of float
        历史能量均值的列表（用于平滑）。
    zcr_history : list of float
        历史过零率均值的列表（用于平滑）。
    alpha : float, default=0.8
        历史阈值与当前统计的加权系数，取值越大越保留历史。
    min_energy_threshold : float, default=1e-6
        能量阈值下限，避免过低。
    max_zcr_threshold : float, default=0.5
        过零率阈值上限，避免过高。

    Returns
    -------
    numpy.ndarray
        形状为 `(num_frames,)` 的布尔数组，`True` 表示语音活动。
    """
    energy = energy.astype(np.float32, copy=False)
    zcr = zcr.astype(np.float32, copy=False)

    # 当前统计
    cur_energy_mean = float(np.mean(energy)) if energy.size else 0.0
    cur_zcr_mean = float(np.mean(zcr)) if zcr.size else 0.0

    # 历史统计（若不存在则使用当前）
    hist_energy = float(np.mean(energy_history)) if energy_history else cur_energy_mean
    hist_zcr = float(np.mean(zcr_history)) if zcr_history else cur_zcr_mean

    # 自适应阈值（指数加权）
    energy_th = max(min_energy_threshold, alpha * hist_energy + (1 - alpha) * cur_energy_mean)
    zcr_th = min(max_zcr_threshold, alpha * hist_zcr + (1 - alpha) * cur_zcr_mean)

    # 历史项目的判定：能量高且 ZCR 高判为语音
    vad = (energy > energy_th) & (zcr > zcr_th)
    return vad.astype(bool)