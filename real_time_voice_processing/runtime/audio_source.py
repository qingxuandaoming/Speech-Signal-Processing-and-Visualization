#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
音频源接口抽象（AudioSource）

提供统一的音频数据读取接口，便于在运行时引擎中替换底层采集实现：
 - PyAudioSource：从系统麦克风实时读取
 - FileAudioSource：从音频文件顺序读取（支持多格式，sf/audioread）
 - PlaylistAudioSource：顺序读取目录/列表中的多个文件

接口约定：
 - open()：打开资源
 - read(num_frames) -> numpy.ndarray[int16]：读取指定样本数；如到达 EOF 可返回空数组
 - close()：关闭资源
 - sample_rate / channels：属性，指示采样率与声道数
"""

from __future__ import annotations

from typing import Optional, Sequence
import numpy as np
from math import gcd as _gcd


# 支持的文件扩展名（大小写不敏感）
SUPPORTED_EXTENSIONS = {
    ".wav", ".flac", ".ogg", ".oga", ".aiff", ".aif", ".mp3", ".m4a", ".aac", ".wma"
}


class AudioSource:
    """音频源基类接口。"""

    sample_rate: int
    channels: int

    def open(self) -> None:
        raise NotImplementedError

    def read(self, num_frames: int) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class PyAudioSource(AudioSource):
    """
    基于 PyAudio 的音频源实现。

    Notes
    -----
    - 依赖 `pyaudio` 包；若未安装或设备不可用，打开时会抛出异常。
    - 返回的数组类型为 `int16`，与引擎处理链一致。
    """

    def __init__(self, sample_rate: int, channels: int, format_const: int, frames_per_buffer: int) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self._format = format_const
        self._fpb = frames_per_buffer
        self._pyaudio = None
        self._stream = None

    def open(self) -> None:
        import pyaudio  # 局部导入，降低模块级依赖

        self._pyaudio = pyaudio.PyAudio()
        self._stream = self._pyaudio.open(
            format=self._format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self._fpb,
        )

    def read(self, num_frames: int) -> np.ndarray:
        assert self._stream is not None, "PyAudioSource 未打开"
        data = self._stream.read(num_frames, exception_on_overflow=False)
        return np.frombuffer(data, dtype=np.int16)

    def close(self) -> None:
        try:
            if self._stream:
                self._stream.stop_stream()
                self._stream.close()
        finally:
            if self._pyaudio:
                self._pyaudio.terminate()
            self._stream = None
            self._pyaudio = None


class FileAudioSource(AudioSource):
    """
    基于文件的音频源实现。

    优先使用 `soundfile` 读取（支持 WAV/FLAC/OGG/AIFF 等），
    若格式不被 libsndfile 支持（如 MP3/M4A），则回退到 `audioread` 解码。

    Parameters
    ----------
    file_path : str
        输入音频文件路径。
    sample_rate : int | None
        若提供则在读取后执行重采样（`scipy.signal.resample_poly`），用于与运行时处理链对齐。
        若为 None 则不重采样，直接返回文件采样率。

    Notes
    -----
    - 依赖 `soundfile`（优先）与 `audioread`（回退，支持 MP3/M4A 等）。
    - 返回的数组类型为 `int16`（单声道）。
    - 重采样采用 `resample_poly`，质量较好且性能可接受。
    """

    def __init__(self, file_path: str, sample_rate: Optional[int] = None) -> None:
        self._file_path = file_path
        self._sf = None
        self._target_sr = sample_rate
        # 在未 open 时，如果提供了目标采样率，则先暴露该值，避免运行时初始化为 0
        self.sample_rate = int(sample_rate or 0)
        self.channels = 1
        # 内部缓冲（当使用整文件解码时）
        self._pcm_array: Optional[np.ndarray] = None
        self._pos = 0
        self._using_sf_stream = False
        # EOF 状态
        self.exhausted: bool = False

    def open(self) -> None:
        self.exhausted = False
        # 优先尝试 soundfile 读取
        try:
            import soundfile as sf  # 局部导入，降低模块级依赖
            # 使用顶层 API 一次性读取，便于统一重采样与下混为单声道
            data, sr = sf.read(self._file_path, dtype="int16", always_2d=False)
            arr = np.asarray(data, dtype=np.int16)
            # 下混为单声道
            if arr.ndim == 2 and arr.shape[1] > 1:
                arr = arr.mean(axis=1).astype(np.int16)
            self.sample_rate = int(sr)
            self.channels = 1
            # 重采样到目标采样率（若指定且不同）
            if self._target_sr and int(self._target_sr) != self.sample_rate:
                arr = _resample_to(arr, self.sample_rate, int(self._target_sr))
                self.sample_rate = int(self._target_sr)
            self._pcm_array = arr
            self._pos = 0
            self._using_sf_stream = False
            return
        except Exception:
            self._pcm_array = None
            self._using_sf_stream = False
            # 回退到 audioread（支持 MP3/M4A 等）
            pass

        # 使用 audioread 解码为 PCM 并加载到内存
        import audioread  # type: ignore
        pcm_list: list[np.ndarray] = []
        with audioread.audio_open(self._file_path) as f:
            sr = int(getattr(f, "samplerate", 0) or 0)
            ch = int(getattr(f, "channels", 1) or 1)
            for buf in f:
                # audioread 输出通常为 16-bit PCM
                pcm_list.append(np.frombuffer(buf, dtype=np.int16))
        if len(pcm_list) == 0:
            self._pcm_array = np.array([], dtype=np.int16)
            self.sample_rate = int(self._target_sr or sr or 0)
            self.channels = 1
        else:
            arr = np.concatenate(pcm_list)
            if ch > 1:
                # 交错布局：LRLR...，下混为单声道
                arr = arr.reshape(-1, ch)[:, 0]
            self.sample_rate = int(sr) if sr else int(self._target_sr or 0)
            self.channels = 1
            if self._target_sr and int(self._target_sr) != self.sample_rate:
                arr = _resample_to(arr, self.sample_rate, int(self._target_sr))
                self.sample_rate = int(self._target_sr)
            self._pcm_array = arr
        self._pos = 0
        self._using_sf_stream = False

    def read(self, num_frames: int) -> np.ndarray:
        # 内存缓冲读取（推荐路径）
        if self._pcm_array is not None:
            start = self._pos
            end = min(start + int(num_frames), len(self._pcm_array))
            chunk = self._pcm_array[start:end]
            self._pos = end
            if self._pos >= len(self._pcm_array):
                self.exhausted = True
            return chunk.astype(np.int16, copy=False)

        # 极端情况下才会走到这里（例如未来切换为流式 sf.SoundFile）
        if self._sf is None:
            return np.array([], dtype=np.int16)
        import soundfile as sf
        data = self._sf.read(num_frames, dtype="int16", always_2d=False)
        if data is None:
            self.exhausted = True
            return np.array([], dtype=np.int16)
        arr = np.array(data, dtype=np.int16)
        if arr.ndim == 2 and arr.shape[1] > 1:
            arr = arr.mean(axis=1).astype(np.int16)
        if self._target_sr and int(self._target_sr) != self.sample_rate:
            arr = _resample_to(arr, self.sample_rate, int(self._target_sr))
        return arr

    def close(self) -> None:
        try:
            if self._sf:
                self._sf.close()
        finally:
            self._sf = None
            self._pcm_array = None
            self._pos = 0
            self.exhausted = True


class PlaylistAudioSource(AudioSource):
    """
    播放列表音频源：顺序读取多个文件，作为一个连续流提供给运行时引擎。

    Parameters
    ----------
    file_paths : Sequence[str]
        要读取的文件路径列表。
    sample_rate : int | None
        目标采样率（若为 None 则使用文件本身采样率；推荐与 Config.SAMPLE_RATE 一致）。
    """

    def __init__(self, file_paths: Sequence[str], sample_rate: Optional[int] = None) -> None:
        self._paths = list(file_paths)
        self._target_sr = sample_rate
        self._current: Optional[FileAudioSource] = None
        self._index = 0
        self.sample_rate = int(sample_rate or 0)
        self.channels = 1
        self.exhausted: bool = False

    def open(self) -> None:
        self._index = 0
        self.exhausted = False
        self._open_current()

    def _open_current(self) -> None:
        if self._index >= len(self._paths):
            self._current = None
            self.exhausted = True
            return
        src = FileAudioSource(self._paths[self._index], sample_rate=self._target_sr)
        src.open()
        # 统一采样率与声道数
        self.sample_rate = int(src.sample_rate or (self._target_sr or 0))
        self.channels = 1
        self._current = src

    def read(self, num_frames: int) -> np.ndarray:
        if self._current is None:
            self.exhausted = True
            return np.array([], dtype=np.int16)
        chunk = self._current.read(num_frames)
        if chunk is None or len(chunk) == 0:
            # 切换到下一个文件
            if self._current:
                self._current.close()
            self._index += 1
            self._open_current()
            if self._current is None:
                self.exhausted = True
                return np.array([], dtype=np.int16)
            chunk = self._current.read(num_frames)
        return chunk

    def close(self) -> None:
        if self._current:
            self._current.close()
        self._current = None
        self._index = 0
        self.exhausted = True


def _resample_to(arr: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """使用 polyphase 方法重采样至目标采样率，并返回 int16。"""
    import scipy.signal as sps  # 局部导入减少模块级依赖
    if src_sr == dst_sr:
        return arr.astype(np.int16, copy=False)
    # 规范化为浮点进行滤波
    x = arr.astype(np.float32)
    g = _gcd(int(src_sr), int(dst_sr))
    up = int(dst_sr // g)
    down = int(src_sr // g)
    y = sps.resample_poly(x, up=up, down=down)
    # 裁剪并转换类型
    y = np.clip(y, -32768.0, 32767.0).astype(np.int16)
    return y