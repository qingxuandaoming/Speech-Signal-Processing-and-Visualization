#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
窗函数生成（Windows）

提供常用窗函数：Hamming、Hanning 与矩形窗。

References
----------
- Oppenheim, A. V., & Schafer, R. W. (2009). Discrete-time signal processing.
"""

import numpy as np


def hamming_window(length: int) -> np.ndarray:
    """
    生成汉明窗（Hamming window）。

    Parameters
    ----------
    length : int
        窗长（样本点数）。

    Returns
    -------
    numpy.ndarray
        长度为 `length` 的窗函数数组，峰值约为 1。
    """
    if length <= 0:
        return np.array([], dtype=np.float32)
    return (0.54 - 0.46 * np.cos(2 * np.pi * np.arange(length) / (length - 1))).astype(
        np.float32
    )


def hanning_window(length: int) -> np.ndarray:
    """
    生成汉宁窗（Hann window）。

    Parameters
    ----------
    length : int
        窗长（样本点数）。

    Returns
    -------
    numpy.ndarray
        长度为 `length` 的窗函数数组，峰值约为 1。
    """
    if length <= 0:
        return np.array([], dtype=np.float32)
    return (0.5 * (1 - np.cos(2 * np.pi * np.arange(length) / (length - 1)))).astype(
        np.float32
    )


def rectangular_window(length: int) -> np.ndarray:
    """
    生成矩形窗（Rectangular window）。

    Parameters
    ----------
    length : int
        窗长（样本点数）。

    Returns
    -------
    numpy.ndarray
        长度为 `length` 的窗函数数组，所有元素为 1。
    """
    if length <= 0:
        return np.array([], dtype=np.float32)
    return np.ones(length, dtype=np.float32)