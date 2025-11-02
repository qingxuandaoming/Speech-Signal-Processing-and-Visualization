#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from typing import Any, Optional
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from real_time_voice_processing.config import Config
from .styles import build_stylesheet, DEFAULT_PALETTE
from .controls import ControlsMixin
from .handlers import EventHandlersMixin

"""
UI for real-time voice processing visualization.

This module provides a PyQtGraph-based GUI to visualize real-time audio
waveform, short-time energy, zero-crossing rate (ZCR), and voice activity
detector (VAD) results. It also offers controls to select input sources,
start/stop processing, and save processed data.

Notes
-----
- The GUI is designed to work with a runtime engine that exposes the
  following methods: `start()`, `stop()`, `get_recent_audio()`,
  `get_recent_processed()`, `save_data()`, and optionally
  `set_audio_source(source, auto_stop_on_eof=True)`.
- Styling leverages a custom palette and Qt stylesheets.
"""


class VisualizationUI(QtCore.QObject, ControlsMixin, EventHandlersMixin):
    """
    可视化界面（VisualizationUI）。

    使用 `pyqtgraph` 展示实时音频波形、短时能量、过零率与语音活动检测结果，
    并提供开始、停止与保存数据的基本交互控件。

    Parameters
    ----------
    runtime : Any
        运行时引擎实例，需实现 `start()`、`stop()`、`get_recent_audio()`、
        `get_recent_processed()` 与 `save_data()` 方法，且可选实现
        `set_audio_source(source, auto_stop_on_eof=True)`。
    title : str, optional
        窗口标题，默认 "实时语音信号处理系统"。
    """

    def __init__(self, runtime: Any, title: str = "实时语音信号处理系统") -> None:
        """初始化界面并连接运行时引擎。

        Parameters
        ----------
        runtime : Any
            运行时引擎实例，需提供实时音频获取与处理接口。
        title : str, default "实时语音信号处理系统"
            窗口标题。

        Returns
        -------
        None
        """
        super().__init__()  # 调用QObject的构造函数
        self.runtime = runtime
        self.app = QtWidgets.QApplication(sys.argv)

        # 主题配色（与用户提供色板一致）
        self.palette = DEFAULT_PALETTE.copy()

        # 应用到 pyqtgraph 全局配置
        pg.setConfigOption('background', self.palette["DARK"])
        pg.setConfigOption('foreground', self.palette["FG"])

        # 应用基础样式（按钮等）
        self.app.setStyleSheet(build_stylesheet(self.palette))

        # 创建窗口并置顶
        self.win = pg.GraphicsLayoutWidget(show=False, title=title)
        try:
            self.win.setWindowTitle(title)
        except Exception:
            # 某些环境下 GraphicsLayoutWidget 可能不支持设置标题；忽略该异常
            pass
        self.win.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)
        self.win.resize(1200, 840)
        self.win.show()

        # 内部状态
        self._selected_file_path: Optional[str] = None
        self._selected_dir_path: Optional[str] = None
        self._done_prompt_shown: bool = False
        
        # 自动坐标范围匹配相关状态
        self._auto_range_enabled: bool = True
        self._range_history = {
            'energy': {'min': [], 'max': []},
            'zcr': {'min': [], 'max': []},
            'vad': {'min': [], 'max': []},
            'audio': {'min': [], 'max': []}
        }
        self._range_buffer_size: int = 10  # 保持最近10次的范围数据用于平滑

        # 构建控件并启动定时刷新
        self._init_ui()
        self._init_timer()

    def _calculate_optimal_range(self, data: np.ndarray, data_type: str) -> tuple[float, float]:
        """计算数据的最优显示范围。

        Parameters
        ----------
        data : np.ndarray
            输入数据数组
        data_type : str
            数据类型，可选值：'energy', 'zcr', 'vad', 'audio'

        Returns
        -------
        tuple[float, float]
            最优的 (min_range, max_range)
        """
        if len(data) == 0:
            # 返回默认范围
            defaults = {
                'energy': (0, 1e10),
                'zcr': (0, 0.5),
                'vad': (-0.1, 1.1),
                'audio': (-32768, 32768)
            }
            return defaults.get(data_type, (0, 1))

        data_min = float(np.min(data))
        data_max = float(np.max(data))

        # 更新历史记录
        if data_type in self._range_history:
            self._range_history[data_type]['min'].append(data_min)
            self._range_history[data_type]['max'].append(data_max)

            # 保持缓冲区大小
            if len(self._range_history[data_type]['min']) > self._range_buffer_size:
                self._range_history[data_type]['min'].pop(0)
                self._range_history[data_type]['max'].pop(0)

            # 使用历史数据计算平滑范围
            hist_min = np.min(self._range_history[data_type]['min'])
            hist_max = np.max(self._range_history[data_type]['max'])
        else:
            hist_min, hist_max = data_min, data_max

        # 根据数据类型调整范围策略
        if data_type == 'energy':
            # 能量数据：使用对数缩放思想，添加适当的上下边距
            if hist_max > 0:
                range_span = hist_max - hist_min
                margin = max(range_span * 0.1, hist_max * 0.05)
                min_range = max(0, hist_min - margin)
                max_range = hist_max + margin
            else:
                min_range, max_range = 0, 100

        elif data_type == 'zcr':
            # 过零率：通常在0-1之间，添加小边距
            range_span = hist_max - hist_min
            margin = max(range_span * 0.1, 0.02)
            min_range = max(0, hist_min - margin)
            max_range = min(1.0, hist_max + margin)

        elif data_type == 'vad':
            # VAD：二值数据，保持固定范围但可微调
            if hist_min >= 0 and hist_max <= 1:
                min_range = -0.1
                max_range = 1.1
            else:
                margin = max((hist_max - hist_min) * 0.1, 0.1)
                min_range = hist_min - margin
                max_range = hist_max + margin

        elif data_type == 'audio':
            # 音频数据：根据实际幅度调整
            range_span = hist_max - hist_min
            margin = max(range_span * 0.1, max(abs(hist_min), abs(hist_max)) * 0.1)
            min_range = hist_min - margin
            max_range = hist_max + margin

        else:
            # 默认策略
            range_span = hist_max - hist_min
            margin = range_span * 0.1 if range_span > 0 else 1
            min_range = hist_min - margin
            max_range = hist_max + margin

        return min_range, max_range

    def _update_plot_ranges(self, recent_audio: np.ndarray, energies: np.ndarray,
                             zcrs: np.ndarray, vads: np.ndarray) -> None:
        """更新所有图表的显示范围。

        Parameters
        ----------
        recent_audio : np.ndarray
            最新音频数据
        energies : np.ndarray
            能量数据
        zcrs : np.ndarray
            过零率数据
        vads : np.ndarray
            VAD数据
        """
        if not self._auto_range_enabled:
            return

        try:
            # 更新音频波形范围
            if len(recent_audio) > 0:
                min_range, max_range = self._calculate_optimal_range(recent_audio, 'audio')
                self.waveform_plot.setYRange(min_range, max_range, padding=0)

            # 更新能量范围
            if len(energies) > 0:
                min_range, max_range = self._calculate_optimal_range(energies, 'energy')
                self.energy_plot.setYRange(min_range, max_range, padding=0)

            # 更新过零率范围
            if len(zcrs) > 0:
                min_range, max_range = self._calculate_optimal_range(zcrs, 'zcr')
                self.zcr_plot.setYRange(min_range, max_range, padding=0)

            # 更新VAD范围
            if len(vads) > 0:
                min_range, max_range = self._calculate_optimal_range(vads, 'vad')
                self.vad_plot.setYRange(min_range, max_range, padding=0)

        except Exception as e:
            # 如果自动范围调整失败，记录错误但不中断程序
            print(f"自动范围调整失败: {e}")

    def _update_status(self):
        """刷新状态标签与控件启用状态。

        Returns
        -------
        None
        """
        self._refresh_controls()

    def _init_timer(self):
        """初始化界面定时器，用于周期刷新图形。

        Returns
        -------
        None
        """
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_plots)
        self.timer.start(int(Config.PLOT_UPDATE_INTERVAL))

    def _update_plots(self):
        """刷新绘图数据。

        从 `runtime` 获取最新音频与特征并更新曲线。包括：波形、短时能量、
        过零率与 VAD。同时应用自动坐标范围匹配。

        Returns
        -------
        None
        """
        try:
            recent_audio = self.runtime.get_recent_audio()
            energies, zcrs, vads = self.runtime.get_recent_processed()
            
            # 应用自动坐标范围匹配
            self._update_plot_ranges(recent_audio, energies, zcrs, vads)
            
            # 更新曲线数据
            if len(recent_audio) > 0:
                self.waveform_curve.setData(recent_audio)

            if len(energies) > 0:
                x_data = np.arange(len(energies))
                self.energy_curve.setData(x_data, energies)
                self.zcr_curve.setData(x_data, zcrs)
                self.vad_curve.setData(x_data, vads)
                # 更新进度（播放列表时显示当前文件序号）
                self._update_progress()
                # 更新结果摘要
                try:
                    energy_mean = float(np.mean(energies))
                    zcr_mean = float(np.mean(zcrs))
                    vad_ratio = float(np.mean(vads)) if len(vads) > 0 else 0.0
                    self.result_label.setText(
                        f"结果：能量均值 {energy_mean:.1f}，ZCR 均值 {zcr_mean:.3f}，语音占比 {vad_ratio*100:.1f}%"
                    )
                except Exception:
                    pass

            # 测试完成后提示继续或关闭（文件或播放列表模式）
            src = getattr(self.runtime, 'audio_source', None)
            if (not getattr(self.runtime, 'is_running', False)) and getattr(src, 'exhausted', False):
                if not self._done_prompt_shown:
                    self._done_prompt_shown = True
                    self._show_done_prompt()
            self._update_status()
        except KeyboardInterrupt:
            # 在控制台中中断程序时，避免抛出异常打断 UI 退出流程
            pass

    def run(self):
        """运行图形界面事件循环。

        Returns
        -------
        None
        """
        sys.exit(self.app.exec())

    # 事件与控件构建由 mixin 提供：ControlsMixin 与 EventHandlersMixin


if __name__ == '__main__':
    """主入口点，启动可视化界面。"""
    try:
        from real_time_voice_processing.runtime.engine import AudioRuntime
        
        # 创建运行时引擎
        runtime = AudioRuntime()
        
        # 创建并显示可视化界面（VisualizationUI会自己创建QApplication）
        ui = VisualizationUI(runtime)
        # 窗口已经在构造函数中显示了，不需要再调用show()
        
        # 运行应用程序
        sys.exit(ui.app.exec())
        
    except Exception as e:
        print(f"启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)