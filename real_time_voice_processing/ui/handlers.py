#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
事件处理（EventHandlersMixin）
============================

该模块提供用于处理界面交互事件与控件状态更新的混入类。包含来源模式切换、
测试范围切换、文件/目录选择、开始/停止/保存、自动坐标范围匹配开关、
弹出列表选择等逻辑。

Notes
-----
- 该混入类依赖于宿主类提供的属性与方法：
  - 运行时：`self.runtime`，包含 `start()`、`stop()`、`get_recent_processed()` 等。
  - 绘图与控件：`self.waveform_plot`、`self.energy_plot`、`self.zcr_plot`、`self.vad_plot` 等。
  - 状态：`self._range_history`、`self._auto_range_enabled`、`self._selected_file_path`、`self._selected_dir_path`。
  - 辅助方法：`self._update_status()`。
"""

import os
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from real_time_voice_processing.config import Config
from real_time_voice_processing.runtime.audio_source import FileAudioSource, PlaylistAudioSource
from .file_utils import default_audio_dir, collect_audio_files


class EventHandlersMixin:
    """事件处理混入类。"""

    def _on_auto_range_toggled(self, enabled: bool) -> None:
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

    def _reset_to_default_ranges(self) -> None:
        """重置所有图表到默认的固定坐标范围。"""
        try:
            # 恢复原始固定范围
            self.waveform_plot.setYRange(-32768, 32768)
            self.energy_plot.setYRange(0, 1e10)
            self.zcr_plot.setYRange(0, 0.5)
            self.vad_plot.setYRange(-0.1, 1.1)
        except Exception as e:
            print(f"重置默认范围失败: {e}")

    def _clear_range_history(self) -> None:
        """清空范围历史记录。"""
        for data_type in self._range_history:
            self._range_history[data_type]['min'].clear()
            self._range_history[data_type]['max'].clear()

    # ---------------------- 状态与启用/禁用 ----------------------

    def _populate_default_dir_files(self) -> None:
        """填充默认目录的文件列表到下拉框。"""
        d = default_audio_dir()
        files = collect_audio_files(d)
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

    def _update_status(self) -> None:
        """刷新状态标签与控件启用状态。"""
        self._refresh_controls()

    def _update_progress(self) -> None:
        """更新进度标签（针对播放列表来源）。"""
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

    # ---------------------- 事件处理 ----------------------

    def _on_stop_clicked(self) -> None:
        """停止运行时引擎并更新界面状态。"""
        try:
            self.runtime.stop()
        finally:
            self._update_status()

    def _on_save_clicked(self) -> None:
        """保存当前处理数据并提示结果。"""
        try:
            path = self.runtime.save_data()
            QtWidgets.QMessageBox.information(self.win, "保存完成", f"数据已保存到：{path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self.win, "保存失败", f"{e}")

    def _on_source_mode_changed(self) -> None:
        """响应来源模式切换并更新控件状态。"""
        custom = self.src_custom_radio.isChecked()
        auto = self.src_auto_radio.isChecked()
        # 当切换到自动目录模式且下拉框为空时，尝试填充默认目录文件
        if auto and self.file_combo.count() == 0:
            self._populate_default_dir_files()
        # 使用麦克风时隐藏进度
        if self.src_mic_radio.isChecked():
            self.progress_label.setText("")
        # 根据来源模式设定复选框默认值（仅在未运行时应用）
        try:
            running = bool(getattr(self.runtime, 'is_running', False))
        except Exception:
            running = False
        if not running:
            if self.src_mic_radio.isChecked():
                # 使用麦克风测试：默认不勾选“模拟实时处理”和“到达文件尾自动停止”
                self.simulate_rt_checkbox.setChecked(False)
                self.auto_stop_checkbox.setChecked(False)
        # 切换到“指定路径”后，若尚未选择，立即弹出目录选择对话框
        if custom and (self._selected_dir_path is None and self._selected_file_path is None):
            self._on_choose_dir()
        # 统一刷新控件状态
        self._refresh_controls()

    def _on_test_option_changed(self) -> None:
        """响应测试范围切换并引导有效来源。"""
        # 单测时：指定目录或自动目录均可通过下拉框选择单个文件
        custom = self.src_custom_radio.isChecked()
        auto = self.src_auto_radio.isChecked()
        mic = self.src_mic_radio.isChecked()
        # 根据测试范围设定复选框默认值（仅在未运行时应用）
        try:
            running = bool(getattr(self.runtime, 'is_running', False))
        except Exception:
            running = False
        if not running and self.test_one_radio.isChecked():
            # 勾选“仅测试一个”时：默认不勾选“到达文件尾自动停止”，默认勾选“模拟实时处理”
            self.auto_stop_checkbox.setChecked(False)
            self.simulate_rt_checkbox.setChecked(True)
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

    def _on_choose_file(self) -> None:
        """选择单个音频文件并更新内部状态。"""
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.win, "选择音频文件", default_audio_dir(),
            "音频文件 (*.wav *.flac *.ogg *.oga *.aiff *.aif *.mp3 *.m4a *.aac *.wma)"
        )
        if f:
            self._selected_file_path = f
            self._selected_dir_path = None
            # 统一刷新控件状态
            self._refresh_controls()

    def _on_choose_dir(self) -> None:
        """选择音频目录并填充文件列表。"""
        d = QtWidgets.QFileDialog.getExistingDirectory(self.win, "选择测试目录", default_audio_dir())
        if d:
            self._selected_dir_path = d
            self._selected_file_path = None
            files = collect_audio_files(d)
            self.file_combo.clear()
            for f in files:
                self.file_combo.addItem(os.path.basename(f), f)
            # 统一刷新控件状态
            self._refresh_controls()

    def _on_start_clicked(self) -> None:
        """根据来源模式设置音源并启动处理。"""
        # 按来源模式明确选择音源，避免意外回退到麦克风
        if self.src_mic_radio.isChecked():
            src = None
        elif self.src_auto_radio.isChecked():
            d = default_audio_dir()
            files = collect_audio_files(d)
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
                files = collect_audio_files(self._selected_dir_path)
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

    def _show_done_prompt(self) -> None:
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

    def _show_file_list_popup(self) -> None:
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