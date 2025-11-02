#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from real_time_voice_processing.config import Config
from real_time_voice_processing.runtime.audio_source import FileAudioSource, PlaylistAudioSource, SUPPORTED_EXTENSIONS


class VisualizationUI:
    """
    可视化界面（VisualizationUI）。

    使用 `pyqtgraph` 展示实时音频波形、短时能量、过零率与语音活动检测结果，
    并提供开始、停止与保存数据的基本交互控件。

    Parameters
    ----------
    runtime : object
        运行时引擎实例，需实现 `start()`、`stop()`、`get_recent_audio()`、`get_recent_processed()` 与 `save_data()` 方法。
    title : str, optional
        窗口标题，默认 "实时语音信号处理系统"。
    """

    def __init__(self, runtime, title="实时语音信号处理系统"):
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
        self.win.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)
        self.win.resize(1200, 840)
        self.win.show()

        # 内部状态
        self._selected_file_path = None
        self._selected_dir_path = None
        self._done_prompt_shown = False

        self._init_ui()
        self._init_timer()

    def _init_ui(self):
        """
        初始化图形界面布局与绘图组件。

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
        self.test_all_radio.setChecked(True)

        test_radio_layout = QtWidgets.QHBoxLayout()
        test_radio_layout.addWidget(self.test_all_radio)
        test_radio_layout.addWidget(self.test_one_radio)

        # 单文件选择（当选择目录且仅测试一个时可用）
        self.file_combo = QtWidgets.QComboBox()
        self.file_combo.setEnabled(False)

        # 说明文字与 EOF 自动停止（对文件/播放列表有效）
        self.auto_stop_checkbox = QtWidgets.QCheckBox("到达文件尾自动停止")
        self.auto_stop_checkbox.setChecked(True)
        self.hint_label = QtWidgets.QLabel(
            "提示：可直接使用麦克风进行实时测试；也可指定文件或目录，并选择仅测试一个。"
        )

        self.settings_layout.addLayout(src_radio_layout)
        self.settings_layout.addLayout(choose_layout)
        self.settings_layout.addLayout(test_radio_layout)
        self.settings_layout.addWidget(self.auto_stop_checkbox)
        self.settings_layout.addWidget(self.file_combo)
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
        try:
            running = getattr(self.runtime, 'is_running', False)
            self.status_label.setText(f"状态：{'运行中' if running else '空闲'}")
        except Exception:
            self.status_label.setText("状态：未知")

    def _update_progress(self):
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
        try:
            self.runtime.stop()
        finally:
            self._update_status()

    def _on_save_clicked(self):
        try:
            path = self.runtime.save_data()
            QtWidgets.QMessageBox.information(self.win, "保存完成", f"数据已保存到：{path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self.win, "保存失败", f"{e}")

    def _init_timer(self):
        """
        初始化界面定时器，用于周期刷新图形。

        Returns
        -------
        None
        """
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_plots)
        self.timer.start(int(Config.PLOT_UPDATE_INTERVAL))

    def _update_plots(self):
        """
        刷新绘图数据：从 `runtime` 获取最新音频与特征并更新曲线。

        Returns
        -------
        None
        """
        try:
            recent_audio = self.runtime.get_recent_audio()
            if len(recent_audio) > 0:
                self.waveform_curve.setData(recent_audio)

            energies, zcrs, vads = self.runtime.get_recent_processed()
            if len(energies) > 0:
                x_data = np.arange(len(energies))
                self.energy_curve.setData(x_data, energies)
                self.zcr_curve.setData(x_data, zcrs)
                self.vad_curve.setData(x_data, vads)
                # 有数据后允许保存
                self.save_btn.setEnabled(True)
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
            if not self.runtime.is_running and getattr(self.runtime.audio_source, 'exhausted', False):
                if not self._done_prompt_shown:
                    self._done_prompt_shown = True
                    self._show_done_prompt()
            self._update_status()
        except KeyboardInterrupt:
            # 在控制台中中断程序时，避免抛出异常打断 UI 退出流程
            pass

    def run(self):
        """
        运行图形界面事件循环。

        Returns
        -------
        None
        """
        sys.exit(self.app.exec_())

    # ---------------------- 逻辑与样式辅助 ----------------------
    def _build_stylesheet(self) -> str:
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
        """

    def _default_audio_dir(self) -> str:
        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        d = os.path.join(pkg_dir, "assets", "audio_tests")
        os.makedirs(d, exist_ok=True)
        return d

    def _collect_audio_files(self, directory: str) -> list[str]:
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
        d = self._default_audio_dir()
        files = self._collect_audio_files(d)
        self.file_combo.clear()
        for f in files:
            self.file_combo.addItem(os.path.basename(f), f)

    def _on_source_mode_changed(self):
        custom = self.src_custom_radio.isChecked()
        mic = self.src_mic_radio.isChecked()
        auto = self.src_auto_radio.isChecked()
        self.choose_file_btn.setEnabled(custom)
        self.choose_dir_btn.setEnabled(custom)
        # 单测模式下，自动或指定目录都允许在下拉框选择一个文件
        has_dir = (self._selected_dir_path is not None)
        has_auto_files = (self.file_combo.count() > 0)
        enable_combo = self.test_one_radio.isChecked() and ((custom and has_dir) or (auto and has_auto_files))
        self.file_combo.setEnabled(enable_combo)
        # 使用麦克风时隐藏进度
        if mic:
            self.progress_label.setText("")
        # 切换到“指定路径”后，若尚未选择，立即弹出目录选择对话框
        if custom and (self._selected_dir_path is None and self._selected_file_path is None):
            self._on_choose_dir()

    def _on_test_option_changed(self):
        # 单测时：指定目录或自动目录均可通过下拉框选择单个文件
        custom = self.src_custom_radio.isChecked()
        auto = self.src_auto_radio.isChecked()
        has_dir = (self._selected_dir_path is not None)
        has_auto_files = (self.file_combo.count() > 0)
        enable_combo = self.test_one_radio.isChecked() and ((custom and has_dir) or (auto and has_auto_files))
        self.file_combo.setEnabled(enable_combo)

    def _on_choose_file(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.win, "选择音频文件", self._default_audio_dir(),
            "音频文件 (*.wav *.flac *.ogg *.oga *.aiff *.aif *.mp3 *.m4a *.aac *.wma)"
        )
        if f:
            self._selected_file_path = f
            self._selected_dir_path = None
            self.file_combo.setEnabled(False)

    def _on_choose_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self.win, "选择测试目录", self._default_audio_dir())
        if d:
            self._selected_dir_path = d
            self._selected_file_path = None
            files = self._collect_audio_files(d)
            self.file_combo.clear()
            for f in files:
                self.file_combo.addItem(os.path.basename(f), f)
            self.file_combo.setEnabled(self.test_one_radio.isChecked())

    def _on_start_clicked(self):
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
            self._done_prompt_shown = False
            self.runtime.start()
            self._update_status()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self.win, "错误", f"启动失败：{e}")

    def _show_done_prompt(self):
        btn = QtWidgets.QMessageBox.question(
            self.win,
            "测试完成",
            "测试已完成。是否继续测试？选择“否”将关闭程序。",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes,
        )
        if btn == QtWidgets.QMessageBox.No:
            self.app.quit()