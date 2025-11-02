#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统配置（Config）

集中管理音频、信号处理、特征提取、VAD、可视化、缓冲与线程等参数。

Attributes
----------
Config : type
    包含全部静态配置项的类。
"""

# 安全导入 PyAudio 常量，未安装时回退到等效数值（paInt16=8）
import os
import logging

try:
    import pyaudio as _pyaudio  # type: ignore
    _AUDIO_FORMAT_DEFAULT = _pyaudio.paInt16
except Exception:
    _AUDIO_FORMAT_DEFAULT = 8

class Config:
    """
    系统配置类。

    Attributes
    ----------
    AUDIO_FORMAT : int
        `pyaudio` 的采样格式常量（`paInt16`）。
    CHANNELS : int
        声道数。
    SAMPLE_RATE : int
        采样率（Hz）。
    CHUNK_SIZE : int
        从设备读取的块大小（样本点）。
    FRAME_DURATION : int
        帧长（毫秒）。
    FRAME_SIZE : int
        帧长对应的样本点数。
    HOP_SIZE : int
        帧移（样本点）。
    WINDOW_TYPE : str
        窗函数类型（`"hamming"`、`"hanning"` 或 `"rectangular"`）。
    PREEMPHASIS_ALPHA : float
        预加重系数。
    NUM_MFCC : int
        MFCC 倒谱系数数量。
    MFCC_N_FFT : int
        MFCC 计算的 FFT 点数。
    MEL_FILTERS : int
        Mel 滤波器数量。
    MFCC_LIFTER : int
        MFCC 倒谱升力系数。
    SPECTRAL_ENTROPY_N_FFT : int
        谱熵计算使用的 FFT 点数。
    ENERGY_THRESHOLD : float
        固定阈值 VAD 的能量阈值。
    ZCR_THRESHOLD : float
        固定阈值 VAD 的过零率阈值。
    ADAPTIVE_VAD_HISTORY_MIN : int
        自适应 VAD 的最小历史长度（未在当前实现中强制）。
    ADAPTIVE_VAD_ENERGY_K : float
        历史版本使用的能量平滑系数（兼容参数）。
    ADAPTIVE_VAD_ZCR_K : float
        历史版本使用的 ZCR 平滑系数（兼容参数）。
    PLOT_UPDATE_INTERVAL : int
        界面刷新间隔（毫秒）。
    MAX_DISPLAY_FRAMES : int
        最大显示帧数。
    WAVEFORM_DISPLAY_LENGTH : int
        波形显示的最大样本数。
    AUDIO_BUFFER_SIZE : int
        音频缓冲区最大块数。
    PROCESSED_DATA_BUFFER_SIZE : int
        已处理数据缓冲区的最大帧数。
    THREAD_SLEEP_TIME : float
        处理线程空闲等待时间（秒）。
    SAVE_DIRECTORY : str
        数据文件保存目录。
    """
    
    # 音频参数
    AUDIO_FORMAT = _AUDIO_FORMAT_DEFAULT  # pyaudio.paInt16
    CHANNELS = 1
    SAMPLE_RATE = 16000  # 16kHz 语音处理标准采样率
    CHUNK_SIZE = 1024  # 每次读取的数据块大小
    FRAME_DURATION = 20  # 帧长（毫秒）
    FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)  # 320个样本点
    HOP_SIZE = FRAME_SIZE // 2  # 帧移，50%重叠
    
    # 信号处理参数
    WINDOW_TYPE = 'hamming'  # 窗函数类型
    PREEMPHASIS_ALPHA = 0.97  # 预加重系数
    
    # 频域与特征参数
    NUM_MFCC = 13  # MFCC 倒谱系数数量
    MFCC_N_FFT = 512  # MFCC 计算的 FFT 点数
    MEL_FILTERS = 26  # Mel 滤波器数量
    MFCC_LIFTER = 22  # MFCC 倒谱升力系数
    SPECTRAL_ENTROPY_N_FFT = 512  # 谱熵计算的 FFT 点数
    
    # 语音活动检测参数
    ENERGY_THRESHOLD = 1000  # 能量阈值
    ZCR_THRESHOLD = 0.3  # 过零率阈值（voiced 判定为 ZCR 较低）
    
    # 自适应 VAD 参数
    ADAPTIVE_VAD_HISTORY_MIN = 20
    ADAPTIVE_VAD_ENERGY_K = 3.0
    ADAPTIVE_VAD_ZCR_K = 1.0
    # 复合VAD门控与平滑
    USE_ADAPTIVE_VAD = True  # 在综合判定中合并自适应VAD结果
    SPECTRAL_ENTROPY_VOICE_MAX = 0.65  # 谱熵低于该阈值更可能为语音
    VAD_HANGOVER_ON = 3  # 进入语音后至少保持的帧数
    VAD_RELEASE_OFF = 2  # 退出语音需连续静音帧数
    
    # 可视化参数
    PLOT_UPDATE_INTERVAL = 50  # 界面更新间隔（毫秒）
    MAX_DISPLAY_FRAMES = 100  # 最大显示帧数
    WAVEFORM_DISPLAY_LENGTH = 4096  # 波形图显示长度
    
    # 缓冲区参数
    AUDIO_BUFFER_SIZE = 4  # 音频缓冲区大小
    PROCESSED_DATA_BUFFER_SIZE = 100  # 处理数据缓冲区大小
    
    # 线程参数
    THREAD_SLEEP_TIME = 0.001  # 线程休眠时间（秒）
    
    # 文件保存参数
    SAVE_DIRECTORY = '.'  # 保存目录
    
    # 日志配置
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '[%(asctime)s] %(levelname)s %(name)s: %(message)s'
    LOG_DATEFMT = '%H:%M:%S'
    
    @staticmethod
    def setup_logging() -> None:
        """
        初始化标准日志配置。

        Returns
        -------
        None
        """
        level = getattr(logging, str(Config.LOG_LEVEL).upper(), logging.INFO)
        logging.basicConfig(level=level, format=Config.LOG_FORMAT, datefmt=Config.LOG_DATEFMT)

    @staticmethod
    def print_config():
        """
        打印关键配置信息到标准日志。

        Returns
        -------
        None
        """
        logging.info("%s", "=" * 50)
        logging.info("实时语音信号处理系统 - 配置信息")
        logging.info("%s", "=" * 50)
        logging.info("音频格式: %s (paInt16)", Config.AUDIO_FORMAT)
        logging.info("声道数: %d", Config.CHANNELS)
        logging.info("采样率: %d Hz", Config.SAMPLE_RATE)
        logging.info("块大小: %d", Config.CHUNK_SIZE)
        logging.info("帧长: %d 样本点 (%dms)", Config.FRAME_SIZE, Config.FRAME_DURATION)
        logging.info("帧移: %d 样本点", Config.HOP_SIZE)
        logging.info("窗函数: %s", Config.WINDOW_TYPE)
        logging.info("预加重系数: %.2f", Config.PREEMPHASIS_ALPHA)
        logging.info(
            "MFCC: num=%d, n_fft=%d, mel_filters=%d, lifter=%d",
            Config.NUM_MFCC,
            Config.MFCC_N_FFT,
            Config.MEL_FILTERS,
            Config.MFCC_LIFTER,
        )
        logging.info("谱熵 FFT 点数: %d", Config.SPECTRAL_ENTROPY_N_FFT)
        logging.info("能量阈值: %.2f", Config.ENERGY_THRESHOLD)
        logging.info("过零率阈值: %.3f", Config.ZCR_THRESHOLD)
        logging.info(
            "自适应VAD: history_min=%d, energy_k=%.2f, zcr_k=%.2f",
            Config.ADAPTIVE_VAD_HISTORY_MIN,
            Config.ADAPTIVE_VAD_ENERGY_K,
            Config.ADAPTIVE_VAD_ZCR_K,
        )

    @staticmethod
    def load_from_env(prefix: str = "RTP_") -> None:
        """
        从环境变量加载配置（覆盖同名项）。

        环境变量命名约定：`<prefix><UPPER_NAME>`，例如 `RTP_SAMPLE_RATE`。

        Returns
        -------
        None
        """
        for name, value in os.environ.items():
            if not name.startswith(prefix):
                continue
            key = name[len(prefix) :]
            if not hasattr(Config, key):
                continue
            current = getattr(Config, key)
            try:
                if isinstance(current, bool):
                    casted = value.lower() in {"1", "true", "yes", "on"}
                elif isinstance(current, int):
                    casted = int(value)
                elif isinstance(current, float):
                    casted = float(value)
                else:
                    casted = value
                setattr(Config, key, casted)
            except Exception:
                logging.warning("环境变量 %s=%s 转换失败，保持默认值", name, value)

    @staticmethod
    def load_from_yaml(path: str) -> bool:
        """
        从 YAML 文件加载配置（需安装 `pyyaml`）。

        Parameters
        ----------
        path : str
            YAML 配置路径。

        Returns
        -------
        bool
            是否加载成功。
        """
        try:
            import yaml  # type: ignore
        except Exception:
            logging.warning("未安装 pyyaml，跳过 YAML 配置加载：%s", path)
            return False

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                logging.warning("YAML 格式不正确（应为字典），忽略：%s", path)
                return False
            for key, value in data.items():
                if hasattr(Config, key):
                    setattr(Config, key, value)
            logging.info("已从 YAML 加载配置：%s", path)
            return True
        except Exception as e:
            logging.error("加载 YAML 配置失败：%s (%s)", path, e)
            return False

# 测试配置
if __name__ == "__main__":
    Config.setup_logging()
    Config.print_config()
