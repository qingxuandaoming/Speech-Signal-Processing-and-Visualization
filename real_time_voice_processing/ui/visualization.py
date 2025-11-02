#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from typing import Any, Optional, List
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from real_time_voice_processing.config import Config
from real_time_voice_processing.runtime.audio_source import FileAudioSource, PlaylistAudioSource, SUPPORTED_EXTENSIONS

"""UI for real-time voice processing visualization.

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


class VisualizationUI(QtCore.QObject):
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
        self.palette = {
            "PRIMARY": "#0A4DAA",   # 深蓝
            "ACCENT": "#2EA3F2",    # 亮蓝
            "GREEN": "#5F7865",     # 灰绿
            "GOLD": "#B8842D",      # 金黄
            "BEIGE": "#EACCA8",     # 米色
            "DARK": "#212C28",      # 深灰绿
            "FG": "#E8EAED"         # 前景/文字
        }

        # 应用到 pyqtgraph 全局配置
        pg.setConfigOption('background', self.palette["DARK"])
        pg.setConfigOption('foreground', self.palette["FG"])

        # 应用基础样式（按钮等）
        self.app.setStyleSheet(self._build_stylesheet())

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
                margin = max(range_span * 0.1, hist_max * 0.05)  # 10%范围边距或5%数值边距
                min_range = max(0, hist_min - margin)  # 能量不能为负
                max_range = hist_max + margin
            else:
                min_range, max_range = 0, 100
                
        elif data_type == 'zcr':
            # 过零率：通常在0-1之间，添加小边距
            range_span = hist_max - hist_min
            margin = max(range_span * 0.1, 0.02)  # 10%范围边距或固定0.02边距
            min_range = max(0, hist_min - margin)  # 过零率不能为负
            max_range = min(1.0, hist_max + margin)  # 过零率不超过1
            
        elif data_type == 'vad':
            # VAD：二值数据，保持固定范围但可微调
            if hist_min >= 0 and hist_max <= 1:
                min_range = -0.1  # 稍微低于0以便观察
                max_range = 1.1   # 稍微高于1以便观察
            else:
                # 异常情况，使用数据范围
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
                           zcrs: np.ndarray, vads: np.ndarray):
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

    def _on_auto_range_toggled(self, enabled: bool):
        """响应自动坐标范围匹配开关切换。
        
        Parameters
        ----------
        enabled : bool
            是否启用自动坐标范围匹配
        """
        self._auto_range_enabled = enabled
        
        if not enabled:
            # 禁用自动范围时，恢复到默认固定范围
            self._reset_to_default_ranges()
        else:
            # 启用自动范围时，清空历史记录以重新开始
            self._clear_range_history()

    def _reset_to_default_ranges(self):
        """重置所有图表到默认的固定坐标范围。"""
        try:
            # 恢复原始固定范围
            self.waveform_plot.setYRange(-32768, 32768)
            self.energy_plot.setYRange(0, 1e10)
            self.zcr_plot.setYRange(0, 0.5)
            self.vad_plot.setYRange(-0.1, 1.1)
        except Exception as e:
            print(f"重置默认范围失败: {e}")

    def _clear_range_history(self):
        """清空范围历史记录。"""
        for data_type in self._range_history:
            self._range_history[data_type]['min'].clear()
            self._range_history[data_type]['max'].clear()

    def _init_ui(self):
        """初始化图形界面布局与绘图组件。

        Returns
        -------
        None
        """
        self.waveform_plot = self.win.addPlot(title="实时音频波形", row=0, col=0)
        self.waveform_curve = self.waveform_plot.plot(pen=self.palette["PRIMARY"])
        self.waveform_plot.setYRange(-32768, 32768)
        self.waveform_plot.setXRange(0, Config.WAVEFORM_DISPLAY_LENGTH)
        self.waveform_plot.setLabel("left", "幅度")
        self.waveform_plot.setLabel("bottom", "样本点")

        self.energy_plot = self.win.addPlot(title="短时能量", row=1, col=0)
        self.energy_curve = self.energy_plot.plot(pen=self.palette["GOLD"])
        self.energy_plot.setYRange(0, 1e10)
        self.energy_plot.setXRange(0, Config.MAX_DISPLAY_FRAMES)
        self.energy_plot.setLabel("left", "能量")
        self.energy_plot.setLabel("bottom", "帧数")

        self.zcr_plot = self.win.addPlot(title="过零率", row=2, col=0)
        self.zcr_curve = self.zcr_plot.plot(pen=self.palette["ACCENT"])
        self.zcr_plot.setYRange(0, 0.5)
        self.zcr_plot.setXRange(0, Config.MAX_DISPLAY_FRAMES)
        self.zcr_plot.setLabel("left", "过零率")
        self.zcr_plot.setLabel("bottom", "帧数")

        self.vad_plot = self.win.addPlot(title="语音活动检测", row=3, col=0)
        self.vad_curve = self.vad_plot.plot(pen=self.palette["BEIGE"], fillLevel=0, brush=QtGui.QColor(self.palette["BEIGE"]))
        self.vad_plot.setYRange(-0.1, 1.1)
        self.vad_plot.setXRange(0, Config.MAX_DISPLAY_FRAMES)
        self.vad_plot.setLabel("left", "语音存在")
        self.vad_plot.setLabel("bottom", "帧数")

        # 交互设置区域（选择音源与测试范围）
        self.settings_layout = QtWidgets.QVBoxLayout()
        self.settings_group = QtWidgets.QGroupBox("测试设置")
        self.settings_group.setLayout(self.settings_layout)

        # 源选择
        self.src_mic_radio = QtWidgets.QRadioButton("使用麦克风")
        self.src_auto_radio = QtWidgets.QRadioButton("自动扫描默认目录")
        self.src_custom_radio = QtWidgets.QRadioButton("指定路径（文件或目录）")
        # 明确分组，避免不同区域的单选互相影响
        self.src_group = QtWidgets.QButtonGroup(self.settings_group)
        # 禁用父级的自动互斥，改由分组控制
        for b in (self.src_mic_radio, self.src_auto_radio, self.src_custom_radio):
            b.setAutoExclusive(False)
            self.src_group.addButton(b)
        self.src_group.setExclusive(True)
        self.src_mic_radio.setChecked(True)

        src_radio_layout = QtWidgets.QHBoxLayout()
        src_radio_layout.addWidget(self.src_mic_radio)
        src_radio_layout.addWidget(self.src_auto_radio)
        src_radio_layout.addWidget(self.src_custom_radio)

        # 自定义路径选择控件
        self.choose_file_btn = QtWidgets.QPushButton("选择文件")
        self.choose_dir_btn = QtWidgets.QPushButton("选择目录")
        self.choose_file_btn.setEnabled(False)
        self.choose_dir_btn.setEnabled(False)

        choose_layout = QtWidgets.QHBoxLayout()
        choose_layout.addWidget(self.choose_file_btn)
        choose_layout.addWidget(self.choose_dir_btn)

        # 测试范围
        self.test_all_radio = QtWidgets.QRadioButton("测试全部")
        self.test_one_radio = QtWidgets.QRadioButton("仅测试一个")
        # 独立分组，确保互斥仅发生在测试范围内
        self.test_group = QtWidgets.QButtonGroup(self.settings_group)
        for b in (self.test_all_radio, self.test_one_radio):
            b.setAutoExclusive(False)
            self.test_group.addButton(b)
        self.test_group.setExclusive(True)
        self.test_all_radio.setChecked(True)

        test_radio_layout = QtWidgets.QHBoxLayout()
        test_radio_layout.addWidget(self.test_all_radio)
        test_radio_layout.addWidget(self.test_one_radio)

        # 单文件选择（当选择目录且仅测试一个时可用）
        # 使用自定义实现解决 QGraphicsProxyWidget 环境下弹出问题
        self.file_combo_container = QtWidgets.QWidget()
        self.file_combo_layout = QtWidgets.QHBoxLayout(self.file_combo_container)
        self.file_combo_layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建可编辑的 QComboBox，禁用默认弹出
        self.file_combo = QtWidgets.QComboBox()
        self.file_combo.setEnabled(False)
        self.file_combo.setEditable(True)  # 设为可编辑以显示当前选择
        self.file_combo.lineEdit().setReadOnly(True)  # 但禁止手动编辑
        self.file_combo.setFocusPolicy(QtCore.Qt.StrongFocus)
        
        # 完全禁用默认弹出机制
        self.file_combo.setMaxVisibleItems(0)
        self.file_combo.setStyleSheet("""
            QComboBox {
                combobox-popup: 0;
                background-color: #2b2b2b;
                border: 1px solid #555;
                padding: 4px;
                color: white;
            }
            QComboBox::drop-down {
                width: 20px;
                border: none;
            }
            QComboBox::down-arrow {
                 image: none;
                 border-left: 3px solid transparent;
                 border-right: 3px solid transparent;
                 border-top: 6px solid white;
                 width: 0px;
                 height: 0px;
                 margin-right: 8px;
             }
            QComboBox QAbstractItemView {
                background-color: #2b2b2b;
                color: white;
                selection-background-color: #0078d4;
            }
        """)
        
        # 创建自定义弹出按钮
        self.file_combo_btn = QtWidgets.QPushButton("▼")
        self.file_combo_btn.setFixedSize(24, 24)
        self.file_combo_btn.setEnabled(False)
        self.file_combo_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                border: none;
                color: white;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #888;
            }
        """)
        
        # 创建弹出列表窗口（独立窗口，不受 QGraphicsProxyWidget 影响）
        self.file_list_popup = None
        
        self.file_combo_layout.addWidget(self.file_combo)
        self.file_combo_layout.addWidget(self.file_combo_btn)
        
        # 连接自定义弹出事件
        self.file_combo_btn.clicked.connect(self._show_file_list_popup)

        # 说明文字与 EOF 自动停止（对文件/播放列表有效）
        self.auto_stop_checkbox = QtWidgets.QCheckBox("到达文件尾自动停止")
        self.auto_stop_checkbox.setChecked(True)

        # 模拟实时处理（文件）：开启后文件按采样率节流
        self.simulate_rt_checkbox = QtWidgets.QCheckBox("模拟实时处理（文件）")
        self.simulate_rt_checkbox.setChecked(False)
        
        # 自动校准阈值：按当前音频自适应能量/ZCR/谱熵阈值
        try:
            default_auto_cal = bool(getattr(Config, 'AUTO_CALIBRATE_THRESHOLDS', True))
        except Exception:
            default_auto_cal = True
        self.auto_calibrate_checkbox = QtWidgets.QCheckBox("自动校准阈值（按当前音频自适应）")
        self.auto_calibrate_checkbox.setChecked(default_auto_cal)
        
        # 自动坐标范围匹配开关
        self.auto_range_checkbox = QtWidgets.QCheckBox("自动匹配坐标范围")
        self.auto_range_checkbox.setChecked(True)
        self.auto_range_checkbox.toggled.connect(self._on_auto_range_toggled)
        
        self.hint_label = QtWidgets.QLabel(
            "提示：可直接使用麦克风进行实时测试；也可指定文件或目录，并选择仅测试一个。"
        )

        self.settings_layout.addLayout(src_radio_layout)
        self.settings_layout.addLayout(choose_layout)
        self.settings_layout.addLayout(test_radio_layout)
        # 四个功能选择键水平排列，减少行占用
        options_row_layout = QtWidgets.QHBoxLayout()
        options_row_layout.addWidget(self.auto_stop_checkbox)
        options_row_layout.addWidget(self.simulate_rt_checkbox)
        options_row_layout.addWidget(self.auto_calibrate_checkbox)
        options_row_layout.addWidget(self.auto_range_checkbox)
        options_row_layout.addStretch(1)
        self.settings_layout.addLayout(options_row_layout)
        self.settings_layout.addWidget(self.file_combo_container)  # 使用容器而不是直接的combo
        self.settings_layout.addWidget(self.hint_label)

        # 添加设置区域到图形布局
        proxy_settings = QtWidgets.QGraphicsProxyWidget()
        proxy_settings.setWidget(self.settings_group)
        self.win.addItem(proxy_settings, row=4, col=0)

        # 控制按钮
        self.ctrl_layout = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("开始处理")
        self.stop_btn = QtWidgets.QPushButton("停止处理")
        self.save_btn = QtWidgets.QPushButton("保存数据")
        # 通过 objectName 使样式选择器生效
        self.stop_btn.setObjectName("stop")
        self.save_btn.setObjectName("save")
        self.save_btn.setEnabled(False)
        # 状态与进度显示
        self.status_label = QtWidgets.QLabel("状态：空闲")
        self.progress_label = QtWidgets.QLabel("")
        self.result_label = QtWidgets.QLabel("")

        self.start_btn.clicked.connect(self._on_start_clicked)
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        self.save_btn.clicked.connect(self._on_save_clicked)

        self.ctrl_layout.addWidget(self.start_btn)
        self.ctrl_layout.addWidget(self.stop_btn)
        self.ctrl_layout.addWidget(self.save_btn)
        self.ctrl_layout.addWidget(self.status_label)
        self.ctrl_layout.addWidget(self.progress_label)
        self.ctrl_layout.addWidget(self.result_label)

        self.ctrl_widget = QtWidgets.QWidget()
        self.ctrl_widget.setLayout(self.ctrl_layout)
        proxy = QtWidgets.QGraphicsProxyWidget()
        proxy.setWidget(self.ctrl_widget)
        self.win.addItem(proxy, row=5, col=0)

        # 信号连接（启用/禁用控件逻辑）
        self.src_mic_radio.toggled.connect(self._on_source_mode_changed)
        self.src_auto_radio.toggled.connect(self._on_source_mode_changed)
        self.src_custom_radio.toggled.connect(self._on_source_mode_changed)
        self.choose_file_btn.clicked.connect(self._on_choose_file)
        self.choose_dir_btn.clicked.connect(self._on_choose_dir)
        self.test_all_radio.toggled.connect(self._on_test_option_changed)
        self.test_one_radio.toggled.connect(self._on_test_option_changed)

        # 初始化默认目录文件列表
        self._populate_default_dir_files()
        self._update_status()

    def _update_status(self):
        """刷新状态标签与控件启用状态。

        Returns
        -------
        None
        """
        self._refresh_controls()

    def _update_progress(self):
        """更新进度标签（针对播放列表来源）。

        Returns
        -------
        None
        """
        src = getattr(self.runtime, 'audio_source', None)
        if src is None:
            self.progress_label.setText("")
            return
        total = len(getattr(src, '_paths', []))
        idx = int(getattr(src, '_index', 0))
        if total > 0:
            self.progress_label.setText(f"进度：{idx+1}/{total}")
        else:
            self.progress_label.setText("")

    def _on_stop_clicked(self):
        """停止运行时引擎并更新界面状态。"""
        try:
            self.runtime.stop()
        finally:
            self._update_status()

    def _on_save_clicked(self):
        """保存当前处理数据并提示结果。"""
        try:
            path = self.runtime.save_data()
            QtWidgets.QMessageBox.information(self.win, "保存完成", f"数据已保存到：{path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self.win, "保存失败", f"{e}")

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
        sys.exit(self.app.exec_())

    # ---------------------- 逻辑与样式辅助 ----------------------
    def _build_stylesheet(self) -> str:
        """构建界面样式表。

        Returns
        -------
        str
            Qt 样式表字符串。
        """
        p = self.palette
        return f"""
        QWidget {{ color: {p['FG']}; }}
        QGroupBox {{ border: 1px solid {p['GREEN']}; border-radius: 6px; margin-top: 6px; }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {p['ACCENT']}; }}
        QPushButton {{
            background-color: {p['PRIMARY']};
            color: white;
            border: none;
            border-radius: 6px;
            padding: 8px 14px;
        }}
        QPushButton:hover {{ background-color: {p['ACCENT']}; }}
        QPushButton#stop {{ background-color: {p['GREEN']}; }}
        QPushButton#save {{ background-color: {p['GOLD']}; }}
        QLabel {{ color: {p['FG']}; }}
        QComboBox {{ background: {p['BEIGE']}; color: black; border-radius: 6px; padding: 4px; }}
        QComboBox:disabled {{ background: #C8C8C8; color: #666; }}
        /* 明确单选按钮的选中/未选样式，提升可辨性 */
        QRadioButton {{ color: {p['FG']}; }}
        QRadioButton::indicator {{ width: 16px; height: 16px; }}
        QRadioButton::indicator:unchecked {{ border: 2px solid {p['ACCENT']}; background: transparent; border-radius: 8px; }}
        QRadioButton::indicator:checked {{ background: {p['ACCENT']}; border: 2px solid {p['ACCENT']}; border-radius: 8px; }}
        QRadioButton:disabled {{ color: #8A8F99; }}
        """

    def _default_audio_dir(self) -> str:
        """获取默认音频目录路径。

        Returns
        -------
        str
            默认音频测试目录的绝对路径。
        """
        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        d = os.path.join(pkg_dir, "assets", "audio_tests")
        os.makedirs(d, exist_ok=True)
        return d

    def _collect_audio_files(self, directory: str) -> list[str]:
        """收集目录下支持的音频文件列表。

        Parameters
        ----------
        directory : str
            目标目录路径。

        Returns
        -------
        list[str]
            按文件名排序的音频文件绝对路径列表。
        """
        exts = {e.lower() for e in SUPPORTED_EXTENSIONS}
        files: list[str] = []
        if not os.path.isdir(directory):
            return files
        for name in sorted(os.listdir(directory)):
            path = os.path.join(directory, name)
            if not os.path.isfile(path):
                continue
            _, ext = os.path.splitext(name)
            if ext.lower() in exts:
                files.append(path)
        return files

    def _populate_default_dir_files(self):
        """填充默认目录的文件列表到下拉框。"""
        d = self._default_audio_dir()
        files = self._collect_audio_files(d)
        self.file_combo.clear()
        for f in files:
            self.file_combo.addItem(os.path.basename(f), f)

    def _should_enable_file_combo(self) -> bool:
        """计算文件下拉框的启用条件。

        自动目录（且有文件）或指定目录（已选择）时启用；与测试范围无关；
        麦克风模式禁用。

        Returns
        -------
        bool
            是否应启用文件下拉框。
        """
        custom = self.src_custom_radio.isChecked()
        auto = self.src_auto_radio.isChecked()
        mic = self.src_mic_radio.isChecked()
        has_dir = (self._selected_dir_path is not None)
        has_auto_files = (self.file_combo.count() > 0)
        return (not mic) and ((custom and has_dir) or (auto and has_auto_files))

    def _refresh_controls(self, running: bool | None = None) -> None:
        """刷新控件启用状态。

        单一入口刷新所有控件状态，消除多处重复/冲突的 `setEnabled` 调用。

        Parameters
        ----------
        running : bool or None, optional
            若提供则以该值为准，否则从运行时引擎读取。

        Returns
        -------
        None
        """
        try:
            if running is None:
                running = getattr(self.runtime, 'is_running', False)

            # 状态标签与开始/停止按钮
            self.status_label.setText(f"状态：{'运行中' if running else '空闲'}")
            self.start_btn.setEnabled(not running)
            self.stop_btn.setEnabled(running)

            # 源与测试范围在运行中锁定，避免状态不一致
            lock = running
            self.src_mic_radio.setEnabled(not lock)
            self.src_auto_radio.setEnabled(not lock)
            self.src_custom_radio.setEnabled(not lock)
            self.test_all_radio.setEnabled(not lock)
            self.test_one_radio.setEnabled(not lock)

            # 选择文件/目录按钮：仅在自定义路径且未运行时可用
            custom = self.src_custom_radio.isChecked()
            self.choose_file_btn.setEnabled(custom and (not lock))
            self.choose_dir_btn.setEnabled(custom and (not lock))

            # EOF 自动停止：文件/播放列表有效；麦克风禁用；运行时锁定当前选择
            mic = self.src_mic_radio.isChecked()
            self.auto_stop_checkbox.setEnabled((not mic) and (not lock))

            # 文件下拉：统一规则（允许运行中预选下一项）
            file_combo_enabled = self._should_enable_file_combo()
            self.file_combo.setEnabled(file_combo_enabled)
            self.file_combo_btn.setEnabled(file_combo_enabled)  # 同步控制弹出按钮

            # 保存按钮：仅在有数据时启用
            try:
                energies, _, _ = self.runtime.get_recent_processed()
                self.save_btn.setEnabled(len(energies) > 0)
            except Exception:
                pass
        except Exception:
            self.status_label.setText("状态：未知")

    def _on_source_mode_changed(self):
        """响应来源模式切换并更新控件状态。"""
        custom = self.src_custom_radio.isChecked()
        auto = self.src_auto_radio.isChecked()
        # 当切换到自动目录模式且下拉框为空时，尝试填充默认目录文件
        if auto and self.file_combo.count() == 0:
            self._populate_default_dir_files()
        # 使用麦克风时隐藏进度
        if self.src_mic_radio.isChecked():
            self.progress_label.setText("")
        # 切换到“指定路径”后，若尚未选择，立即弹出目录选择对话框
        if custom and (self._selected_dir_path is None and self._selected_file_path is None):
            self._on_choose_dir()
        # 统一刷新控件状态
        self._refresh_controls()

    def _on_test_option_changed(self):
        """响应测试范围切换并引导有效来源。"""
        # 单测时：指定目录或自动目录均可通过下拉框选择单个文件
        custom = self.src_custom_radio.isChecked()
        auto = self.src_auto_radio.isChecked()
        mic = self.src_mic_radio.isChecked()
        # 若选中“仅测试一个”，但当前为麦克风模式，则优先引导可用来源
        if self.test_one_radio.isChecked() and mic:
            if self.file_combo.count() == 0:
                self._populate_default_dir_files()
            if self.file_combo.count() > 0:
                self.src_auto_radio.setChecked(True)
            else:
                self.src_custom_radio.setChecked(True)
                if self._selected_dir_path is None:
                    self._on_choose_dir()
        # 统一刷新控件状态
        self._refresh_controls()

    def _on_choose_file(self):
        """选择单个音频文件并更新内部状态。"""
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.win, "选择音频文件", self._default_audio_dir(),
            "音频文件 (*.wav *.flac *.ogg *.oga *.aiff *.aif *.mp3 *.m4a *.aac *.wma)"
        )
        if f:
            self._selected_file_path = f
            self._selected_dir_path = None
            # 统一刷新控件状态
            self._refresh_controls()

    def _on_choose_dir(self):
        """选择音频目录并填充文件列表。"""
        d = QtWidgets.QFileDialog.getExistingDirectory(self.win, "选择测试目录", self._default_audio_dir())
        if d:
            self._selected_dir_path = d
            self._selected_file_path = None
            files = self._collect_audio_files(d)
            self.file_combo.clear()
            for f in files:
                self.file_combo.addItem(os.path.basename(f), f)
            # 统一刷新控件状态
            self._refresh_controls()

    def _on_start_clicked(self):
        """根据来源模式设置音源并启动处理。"""
        # 按来源模式明确选择音源，避免意外回退到麦克风
        if self.src_mic_radio.isChecked():
            src = None
        elif self.src_auto_radio.isChecked():
            d = self._default_audio_dir()
            files = self._collect_audio_files(d)
            if len(files) == 0:
                QtWidgets.QMessageBox.warning(self.win, "提示", "默认目录为空，请选择“指定路径”或使用麦克风。")
                return
            if self.test_one_radio.isChecked():
                # 若下拉框有默认目录预填的项目，使用当前项；否则使用第一个
                if self.file_combo.count() > 0:
                    selected = self.file_combo.currentData()
                    files = [selected]
                else:
                    files = [files[0]]
            else:
                # 测试全部：若下拉选择了某项，作为起始文件重排
                if self.file_combo.count() > 0:
                    selected = self.file_combo.currentData()
                    if selected and selected in files:
                        idx = files.index(selected)
                        files = files[idx:] + files[:idx]
            src = PlaylistAudioSource(files, sample_rate=Config.SAMPLE_RATE)
        elif self.src_custom_radio.isChecked():
            # 若尚未选择，先引导选择目录
            if not self._selected_file_path and not self._selected_dir_path:
                self._on_choose_dir()
            if self._selected_file_path:
                src = FileAudioSource(self._selected_file_path, sample_rate=Config.SAMPLE_RATE)
            elif self._selected_dir_path:
                files = self._collect_audio_files(self._selected_dir_path)
                if len(files) == 0:
                    QtWidgets.QMessageBox.warning(self.win, "提示", "所选目录为空，请重新选择。")
                    return
                if self.test_one_radio.isChecked():
                    if self.file_combo.count() == 0:
                        QtWidgets.QMessageBox.warning(self.win, "提示", "请先选择一个文件。")
                        return
                    files = [self.file_combo.currentData()]
                else:
                    # 测试全部：若下拉选择了某项，作为起始文件重排
                    if self.file_combo.count() > 0:
                        selected = self.file_combo.currentData()
                        if selected and selected in files:
                            idx = files.index(selected)
                            files = files[idx:] + files[:idx]
                src = PlaylistAudioSource(files, sample_rate=Config.SAMPLE_RATE)
            else:
                QtWidgets.QMessageBox.warning(self.win, "提示", "未选择文件或目录。")
                return

        # 设置音源并启动
        try:
            # 若引擎支持设置自动 EOF 停止，则开启
            if hasattr(self.runtime, 'set_audio_source'):
                self.runtime.set_audio_source(src, auto_stop_on_eof=self.auto_stop_checkbox.isChecked())
            else:
                # 回退：直接替换属性（不推荐）
                self.runtime.audio_source = src or self.runtime.audio_source
            # 运行前联动配置：是否模拟实时处理文件
            try:
                from real_time_voice_processing.config import Config as _Cfg
                _Cfg.SIMULATE_REALTIME_FILES = bool(self.simulate_rt_checkbox.isChecked())
                _Cfg.AUTO_CALIBRATE_THRESHOLDS = bool(self.auto_calibrate_checkbox.isChecked())
            except Exception:
                pass
            # 模拟实时处理时同步播放
            try:
                if hasattr(self.runtime, 'set_playback_enabled'):
                    self.runtime.set_playback_enabled(bool(self.simulate_rt_checkbox.isChecked()))
            except Exception:
                pass
            self._done_prompt_shown = False
            self.runtime.start()
            self._update_status()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self.win, "错误", f"启动失败：{e}")

    def _show_done_prompt(self):
        """展示测试完成提示并处理用户选择。"""
        btn = QtWidgets.QMessageBox.question(
            self.win,
            "测试完成",
            "测试已完成。是否继续测试？选择\"否\"将关闭程序。",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes,
        )
        if btn == QtWidgets.QMessageBox.No:
            self.app.quit()

    def _show_file_list_popup(self):
        """显示自定义文件列表弹出窗口。"""
        if not self.file_combo.isEnabled() or self.file_combo.count() == 0:
            return
            
        # 如果弹出窗口已存在，先关闭
        if self.file_list_popup is not None:
            self.file_list_popup.close()
            self.file_list_popup = None
            return
            
        # 创建独立的弹出窗口
        self.file_list_popup = QtWidgets.QDialog(self.win)
        self.file_list_popup.setWindowFlags(
            QtCore.Qt.Popup | QtCore.Qt.FramelessWindowHint
        )
        self.file_list_popup.setModal(False)
        
        # 设置弹出窗口样式
        self.file_list_popup.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                border: 1px solid #555;
            }
        """)
        
        # 创建列表控件
        layout = QtWidgets.QVBoxLayout(self.file_list_popup)
        layout.setContentsMargins(0, 0, 0, 0)
        
        list_widget = QtWidgets.QListWidget()
        list_widget.setStyleSheet("""
            QListWidget {
                background-color: #2b2b2b;
                color: white;
                border: none;
                outline: none;
            }
            QListWidget::item {
                padding: 4px;
                border-bottom: 1px solid #444;
            }
            QListWidget::item:hover {
                background-color: #3c3c3c;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
            }
        """)
        
        # 填充列表项
        for i in range(self.file_combo.count()):
            item_text = self.file_combo.itemText(i)
            item_data = self.file_combo.itemData(i)
            list_item = QtWidgets.QListWidgetItem(item_text)
            list_item.setData(QtCore.Qt.UserRole, item_data)
            list_widget.addItem(list_item)
            
            # 设置当前选中项
            if i == self.file_combo.currentIndex():
                list_widget.setCurrentItem(list_item)
        
        layout.addWidget(list_widget)
        
        # 连接选择事件
        def on_item_clicked(item):
            # 更新 combo box 的选择
            for i in range(self.file_combo.count()):
                if self.file_combo.itemData(i) == item.data(QtCore.Qt.UserRole):
                    self.file_combo.setCurrentIndex(i)
                    break
            # 关闭弹出窗口
            self.file_list_popup.close()
            self.file_list_popup = None
            
        list_widget.itemClicked.connect(on_item_clicked)
        
        # 计算弹出位置（相对于主窗口）
        try:
            # 获取 combo 容器在主窗口中的全局位置
            combo_global_pos = self.file_combo_container.mapToGlobal(QtCore.QPoint(0, 0))
            combo_size = self.file_combo_container.size()
            
            # 设置弹出窗口大小和位置
            popup_width = max(200, combo_size.width())
            popup_height = min(200, list_widget.sizeHintForRow(0) * min(8, list_widget.count()) + 10)
            
            self.file_list_popup.resize(popup_width, popup_height)
            self.file_list_popup.move(combo_global_pos.x(), combo_global_pos.y() + combo_size.height())
            
            # 显示弹出窗口
            self.file_list_popup.show()
            list_widget.setFocus()
            
        except Exception as e:
            print(f"弹出窗口定位失败: {e}")
            # 如果定位失败，在鼠标位置显示
            cursor_pos = QtGui.QCursor.pos()
            self.file_list_popup.move(cursor_pos.x(), cursor_pos.y())
            self.file_list_popup.show()
            
        # 设置失去焦点时自动关闭
        def on_focus_out():
            if self.file_list_popup is not None:
                self.file_list_popup.close()
                self.file_list_popup = None
                
        # 使用定时器延迟绑定焦点事件，避免立即触发
        def install_filter():
            if self.file_list_popup is not None:
                self.file_list_popup.installEventFilter(self)
        QtCore.QTimer.singleShot(100, install_filter)
        
    def eventFilter(self, obj, event):
        """事件过滤器，用于处理弹出窗口的焦点丢失。"""
        if obj == self.file_list_popup and event.type() == QtCore.QEvent.WindowDeactivate:
            if self.file_list_popup is not None:
                self.file_list_popup.close()
                self.file_list_popup = None
        return super().eventFilter(obj, event)


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