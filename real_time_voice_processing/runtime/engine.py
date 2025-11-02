#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import threading
import logging
from collections import deque
import numpy as np

from real_time_voice_processing.config import Config
from real_time_voice_processing.signal_processing import SignalProcessing
from real_time_voice_processing.runtime.audio_source import AudioSource, PyAudioSource


logger = logging.getLogger(__name__)


class AudioRuntime:
    """
    运行时引擎（AudioRuntime）。

    负责管理音频采集与信号处理两个线程，并提供数据访问与保存接口。

    Attributes
    ----------
    format : int
        音频采样格式（`pyaudio` 常量）。
    channels : int
        声道数。
    rate : int
        采样率（Hz）。
    chunk : int
        每次从音频设备读取的块大小（样本点）。
    frame_size : int
        分析帧长度（样本点）。
    hop_size : int
        帧移（样本点）。
    window : numpy.ndarray
        当前使用的窗函数（长度为 `frame_size`）。
    energy_threshold : float
        语音检测的能量阈值。
    zcr_threshold : float
        语音检测的过零率阈值。
    audio_buffer : collections.deque
        原始音频缓冲区（最新数据在尾部）。
    processed_data : collections.deque
        已处理特征的缓冲区，元素为字典：`{"energy", "zcr", "vad", "spec_entropy", "vad_adaptive", "mfcc"}`。
    energy_history : collections.deque
        能量历史（用于自适应 VAD）。
    zcr_history : collections.deque
        过零率历史（用于自适应 VAD）。
    is_running : bool
        运行标记。
    audio_thread : threading.Thread | None
        音频采集线程。
    processing_thread : threading.Thread | None
        信号处理线程。
    lock : threading.Lock
        线程间共享数据的互斥锁。
    """

    def __init__(self, audio_source: AudioSource | None = None):
        # 基本参数
        self.format = Config.AUDIO_FORMAT
        self.chunk = Config.CHUNK_SIZE

        # 音频源（可注入实现）
        if audio_source is None:
            # 默认使用系统麦克风作为输入
            audio_source = PyAudioSource(
                sample_rate=Config.SAMPLE_RATE,
                channels=Config.CHANNELS,
                format_const=Config.AUDIO_FORMAT,
                frames_per_buffer=Config.CHUNK_SIZE,
            )
        self.audio_source: AudioSource = audio_source
        self.rate = getattr(audio_source, "sample_rate", Config.SAMPLE_RATE)
        self.channels = getattr(audio_source, "channels", Config.CHANNELS)
        self.frame_size = Config.FRAME_SIZE
        self.hop_size = Config.HOP_SIZE

        # 窗函数
        self.window = SignalProcessing.hamming_window(self.frame_size)

        # 阈值
        self.energy_threshold = Config.ENERGY_THRESHOLD
        self.zcr_threshold = Config.ZCR_THRESHOLD

        # 缓冲区
        self.audio_buffer = deque(maxlen=Config.AUDIO_BUFFER_SIZE)
        # 独立的显示缓冲：避免被处理线程立即消费导致波形不连续
        display_chunks = max(8, int(Config.WAVEFORM_DISPLAY_LENGTH / max(1, Config.CHUNK_SIZE)))
        self.audio_display_buffer = deque(maxlen=display_chunks)
        self.processed_data = deque(maxlen=Config.PROCESSED_DATA_BUFFER_SIZE)
        # 特征历史（用于自适应VAD）
        self.energy_history = deque(maxlen=256)
        self.zcr_history = deque(maxlen=256)

        # 线程控制
        self.is_running = False
        self.audio_thread = None
        self.processing_thread = None
        self.lock = threading.Lock()
        self.last_error = None  # 记录最近发生的异常，便于诊断
        # EOF 自动停止支持（由 UI 控制）
        self.auto_stop_on_eof: bool = False
        # VAD 平滑状态
        self._vad_hold: int = 0
        self._silence_run: int = 0

    def set_audio_source(self, audio_source: AudioSource | None, auto_stop_on_eof: bool = False) -> None:
        """
        设置/更换当前音频源。

        Parameters
        ----------
        audio_source : AudioSource | None
            新的音频源；若为 None 则回退为系统麦克风。
        auto_stop_on_eof : bool
            当文件/播放列表读到 EOF 时是否自动停止运行。
        """
        # 若正在运行，先安全停止
        if self.is_running:
            self.stop()
        if audio_source is None:
            audio_source = PyAudioSource(
                sample_rate=Config.SAMPLE_RATE,
                channels=Config.CHANNELS,
                format_const=Config.AUDIO_FORMAT,
                frames_per_buffer=Config.CHUNK_SIZE,
            )
        self.audio_source = audio_source
        self.rate = getattr(audio_source, "sample_rate", Config.SAMPLE_RATE)
        self.channels = getattr(audio_source, "channels", Config.CHANNELS)
        self.auto_stop_on_eof = bool(auto_stop_on_eof)
        # 重置缓冲区与历史
        self.audio_buffer.clear()
        self.processed_data.clear()
        self.energy_history.clear()
        self.zcr_history.clear()
        self.audio_display_buffer.clear()

    def start(self):
        """
        启动运行时引擎。

        创建并启动音频采集与信号处理线程。若已在运行则忽略。

        Returns
        -------
        None
        """
        if not self.is_running:
            self.is_running = True
            self.audio_thread = threading.Thread(target=self._audio_capture_thread, daemon=True)
            self.processing_thread = threading.Thread(target=self._signal_processing_thread, daemon=True)
            self.audio_thread.start()
            self.processing_thread.start()

    def stop(self):
        """
        停止运行时引擎。

        设置运行标记为 False，并等待采集与处理线程安全退出。

        Returns
        -------
        None
        """
        if self.is_running:
            self.is_running = False
            if self.audio_thread:
                self.audio_thread.join()
            if self.processing_thread:
                self.processing_thread.join()

    def _audio_capture_thread(self):
        """
        音频采集线程函数。

        从系统音频设备连续读取数据并写入 `audio_buffer`。异常发生时尽力释放设备资源。

        Returns
        -------
        None
        """
        # 使用注入的音频源进行读取
        stream_opened = False
        try:
            self.audio_source.open()
            stream_opened = True
            while self.is_running:
                audio_data = self.audio_source.read(self.chunk)
                if audio_data is None or len(audio_data) == 0:
                    # 文件/播放列表源：EOF 检测
                    exhausted = bool(getattr(self.audio_source, "exhausted", False))
                    if exhausted and self.auto_stop_on_eof:
                        # 自动停止并退出采集线程
                        self.is_running = False
                        break
                    time.sleep(Config.THREAD_SLEEP_TIME)
                    continue
                with self.lock:
                    self.audio_buffer.append(audio_data)
                    # 保留一份用于波形显示
                    self.audio_display_buffer.append(np.array(audio_data, copy=True))
        except Exception as e:
            # 避免静默吞掉异常，记录并输出便于定位问题
            self.last_error = e
            logger.exception("音频采集线程异常: %s", e)
        finally:
            try:
                if stream_opened:
                    self.audio_source.close()
            except Exception:
                # 关闭异常不应影响退出
                pass

    def _signal_processing_thread(self):
        """
        信号处理线程函数。

        从 `audio_buffer` 中取数据并按帧处理，计算时域与频域特征、语音活动检测，并写入 `processed_data`。

        Returns
        -------
        None
        """
        overlap_buffer = np.array([], dtype=np.int16)
        sleep_time = Config.THREAD_SLEEP_TIME

        while self.is_running:
            if len(self.audio_buffer) == 0:
                time.sleep(sleep_time)
                continue
            with self.lock:
                audio_data = self.audio_buffer.popleft()
            overlap_buffer = np.concatenate((overlap_buffer, audio_data))

            while len(overlap_buffer) >= self.frame_size:
                frame = overlap_buffer[: self.frame_size].astype(np.float32)
                overlap_buffer = overlap_buffer[self.hop_size :]

                windowed_frame = frame * self.window
                energy = SignalProcessing.calculate_short_time_energy(windowed_frame)
                zcr = SignalProcessing.calculate_zero_crossing_rate(windowed_frame)

                # 频域特征
                spec_entropy = SignalProcessing.calculate_spectral_entropy(
                    windowed_frame, n_fft=Config.SPECTRAL_ENTROPY_N_FFT
                )

                # 基本门控：能量高 + （ZCR 低 或 谱熵低）
                energy_gate = bool(energy > self.energy_threshold)
                zcr_gate = bool(zcr < self.zcr_threshold)
                entropy_gate = bool(spec_entropy < Config.SPECTRAL_ENTROPY_VOICE_MAX)
                vad_initial = bool(energy_gate and (zcr_gate or entropy_gate))

                # 自适应VAD（用于增强稳健性，可选合并）
                vad_adaptive = SignalProcessing.adaptive_voice_activity_detection(
                    energy,
                    zcr,
                    list(self.energy_history),
                    list(self.zcr_history),
                    energy_k=Config.ADAPTIVE_VAD_ENERGY_K,
                    zcr_k=Config.ADAPTIVE_VAD_ZCR_K,
                    min_history=Config.ADAPTIVE_VAD_HISTORY_MIN,
                    fallback_energy_threshold=self.energy_threshold,
                    fallback_zcr_threshold=self.zcr_threshold,
                )
                if Config.USE_ADAPTIVE_VAD:
                    vad_initial = bool(vad_initial or bool(vad_adaptive))

                # 延滞/释出平滑：减少抖动
                if vad_initial:
                    self._vad_hold = max(self._vad_hold, int(Config.VAD_HANGOVER_ON))
                    self._silence_run = 0
                    vad = 1
                else:
                    if self._vad_hold > 0:
                        # 保持一段时间仍视为语音
                        self._vad_hold -= 1
                        vad = 1
                        self._silence_run = 0
                    else:
                        # 需要连续静音帧数后才确认静音
                        self._silence_run += 1
                        vad = 0 if self._silence_run >= int(Config.VAD_RELEASE_OFF) else 1
                mfcc = SignalProcessing.compute_mfcc(
                    windowed_frame,
                    sample_rate=self.rate,
                    num_ceps=Config.NUM_MFCC,
                    n_fft=Config.MFCC_N_FFT,
                    n_filters=Config.MEL_FILTERS,
                    lifter=Config.MFCC_LIFTER,
                    pre_emphasis=None,
                )

                with self.lock:
                    self.energy_history.append(float(energy))
                    self.zcr_history.append(float(zcr))
                    self.processed_data.append(
                        {
                            "energy": float(energy),
                            "zcr": float(zcr),
                            "vad": int(vad),
                            "spec_entropy": float(spec_entropy),
                            "vad_adaptive": int(vad_adaptive),
                            "mfcc": mfcc.tolist(),
                        }
                    )

    def get_recent_audio(self):
        """
        获取最近音频波形片段。

        Returns
        -------
        numpy.ndarray
        一维整型数组，长度不超过 `Config.WAVEFORM_DISPLAY_LENGTH`。
        """
        with self.lock:
            if len(self.audio_display_buffer) == 0:
                return np.array([], dtype=np.int16)
            recent_audio = np.concatenate(list(self.audio_display_buffer))
        length = Config.WAVEFORM_DISPLAY_LENGTH
        if len(recent_audio) > length:
            recent_audio = recent_audio[-length:]
        return recent_audio

    def get_recent_processed(self, max_display=None):
        """
        获取最近处理后的特征。

        Parameters
        ----------
        max_display : int or None, optional
            返回的最大帧数；若为 `None` 则使用 `Config.MAX_DISPLAY_FRAMES`。

        Returns
        -------
        tuple of numpy.ndarray
            `(energies, zcrs, vads)` 三个一维数组。
        """
        if max_display is None:
            max_display = Config.MAX_DISPLAY_FRAMES
        with self.lock:
            if len(self.processed_data) == 0:
                return np.array([]), np.array([]), np.array([])
            energies = [d["energy"] for d in self.processed_data]
            zcrs = [d["zcr"] for d in self.processed_data]
            vads = [d["vad"] for d in self.processed_data]
        if len(energies) > max_display:
            energies = energies[-max_display:]
            zcrs = zcrs[-max_display:]
            vads = vads[-max_display:]
        return np.array(energies), np.array(zcrs), np.array(vads)

    def save_data(self, directory=None):
        """
        保存处理数据到 NPZ 文件。

        Parameters
        ----------
        directory : str or None, optional
            保存目录；若为 `None` 则使用 `Config.SAVE_DIRECTORY`。

        Returns
        -------
        str
            保存的文件路径。
        """
        if directory is None:
            directory = Config.SAVE_DIRECTORY
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{directory}/voice_processing_data_{timestamp}.npz"
        energies, zcrs, vads = self.get_recent_processed(max_display=Config.PROCESSED_DATA_BUFFER_SIZE)
        # 其他可选特征
        with self.lock:
            spec_entropies = [d.get("spec_entropy", np.nan) for d in self.processed_data]
            vads_adaptive = [d.get("vad_adaptive", np.nan) for d in self.processed_data]
        if len(spec_entropies) > Config.PROCESSED_DATA_BUFFER_SIZE:
            spec_entropies = spec_entropies[-Config.PROCESSED_DATA_BUFFER_SIZE:]
            vads_adaptive = vads_adaptive[-Config.PROCESSED_DATA_BUFFER_SIZE:]
        np.savez(
            filename,
            energies=np.array(energies),
            zcrs=np.array(zcrs),
            vads=np.array(vads),
            spec_entropy=np.array(spec_entropies, dtype=np.float32),
            vads_adaptive=np.array(vads_adaptive, dtype=np.float32),
            sample_rate=self.rate,
            frame_size=self.frame_size,
            hop_size=self.hop_size,
        )
        return filename