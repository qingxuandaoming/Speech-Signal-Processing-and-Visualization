#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
控件构建（ControlsMixin）
=======================

该模块提供用于构建可视化界面控件与绘图区域的混入类。将 UI 控件的创建、布局、
样式与信号连接集中在一个位置，便于与事件处理逻辑解耦。

Notes
-----
- 该混入类依赖于宿主类已初始化的属性：
  - `self.win`: `pyqtgraph.GraphicsLayoutWidget` 主窗口/画布。
  - `self.palette`: 颜色字典，用于统一配色。
  - `self._refresh_controls`: 宿主方法，用于刷新控件启用状态。
  - 事件处理方法如 `_on_start_clicked`、`_on_source_mode_changed` 等。
"""

from typing import Any
import os
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from real_time_voice_processing.config import Config
from .file_utils import default_audio_dir, collect_audio_files


class ControlsMixin:
    """
    控件构建混入类。

    提供 `_init_ui` 方法以在主窗口中创建绘图区域与交互控件，并完成信号连接。
    """

    def _init_ui(self) -> None:
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
        # 默认：麦克风模式下不相关且不勾选；初始为麦克风
        self.auto_stop_checkbox.setChecked(False)

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

        # 初始化默认目录文件列表与状态
        self._populate_default_dir_files()
        self._update_status()