#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from typing import List
from real_time_voice_processing.config import Config
from real_time_voice_processing.runtime.engine import AudioRuntime
from real_time_voice_processing.runtime.audio_source import FileAudioSource, PlaylistAudioSource, SUPPORTED_EXTENSIONS
"""
注意：Qt 插件路径在部分 Windows 环境下可能未自动发现，导致
"qt.qpa.plugin: Could not find the Qt platform plugin \"windows\"" 错误。
为提升稳健性，我们在运行时设置 QT_QPA_PLATFORM_PLUGIN_PATH。
"""



def main():
    # 初始化日志并加载配置（环境变量优先，YAML 可选）
    Config.setup_logging()
    yaml_path = os.environ.get("RTP_CONFIG_YAML")
    if yaml_path:
        Config.load_from_yaml(yaml_path)
    Config.load_from_env(prefix="RTP_")

    # 强制 pyqtgraph 使用 PySide6（在部分 Windows 环境下更稳定）
    os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

    # 可选：文件/目录音频测试源（适合集成测试/演示）
    input_file = os.environ.get("RTP_INPUT_FILE")
    input_dir = os.environ.get("RTP_INPUT_DIR")

    audio_source = None
    if input_file:
        audio_source = FileAudioSource(input_file, sample_rate=Config.SAMPLE_RATE)
    elif input_dir:
        files = _collect_audio_files(input_dir)
        if files:
            audio_source = PlaylistAudioSource(files, sample_rate=Config.SAMPLE_RATE)
    # 未设置环境变量时不进行终端交互，交互逻辑在 UI 中完成

    # 确保 Qt 插件路径
    _ensure_qt_plugins_env()

    # 延迟导入 UI，避免在模块导入阶段初始化 Qt
    from real_time_voice_processing.ui.visualization import VisualizationUI

    runtime = AudioRuntime(audio_source=audio_source)
    ui = VisualizationUI(runtime)
    ui.run()


 


def _default_audio_dir() -> str:
    """默认测试音频目录：real_time_voice_processing/assets/audio_tests"""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    d = os.path.join(pkg_dir, "assets", "audio_tests")
    os.makedirs(d, exist_ok=True)
    return d


def _collect_audio_files(directory: str) -> List[str]:
    exts = {e.lower() for e in SUPPORTED_EXTENSIONS}
    files: List[str] = []
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


def _interactive_choose_audio_source():
    """
    在启动前进行简单的交互式选择：
    - 指定路径（文件/目录）或自动测试（默认选择自动测试，扫描默认目录）
    - 若为目录：选择测试全部或测试一个（默认全部）
    返回构造好的音频源，若无可用文件则返回 None（使用麦克风）。
    """
    try:
        default_dir = _default_audio_dir()
        print("\n[音频测试] 选择音频来源：")
        print("1) 自动扫描默认目录: ", default_dir)
        print("2) 指定路径（文件或目录）")
        choice = input("请输入 1/2 [默认1]: ").strip()
        if choice not in {"1", "2"}:
            choice = "1"

        if choice == "2":
            path = input("请输入文件或目录路径: ").strip().strip('"')
            if os.path.isdir(path):
                files = _collect_audio_files(path)
                if not files:
                    print("未在目录中找到支持的音频文件，回退到麦克风。")
                    return None
                print(f"在目录中找到 {len(files)} 个音频文件。")
                print("1) 测试全部  2) 仅测试一个")
                sub = input("请输入 1/2 [默认1]: ").strip()
                if sub == "2":
                    for i, f in enumerate(files, start=1):
                        print(f"{i}. {os.path.basename(f)}")
                    try:
                        idx = int(input("选择编号 [默认1]: ") or "1")
                    except Exception:
                        idx = 1
                    idx = max(1, min(idx, len(files)))
                    files = [files[idx - 1]]
                return PlaylistAudioSource(files, sample_rate=Config.SAMPLE_RATE)
            elif os.path.isfile(path):
                return FileAudioSource(path, sample_rate=Config.SAMPLE_RATE)
            else:
                print("路径无效，回退到默认目录自动扫描。")

        # 自动扫描默认目录
        files = _collect_audio_files(default_dir)
        if not files:
            print("默认目录为空，未找到音频文件。将使用麦克风进行实时测试。")
            return None
        print(f"在默认目录中找到 {len(files)} 个音频文件。")
        print("1) 测试全部  2) 仅测试一个")
        sub = input("请输入 1/2 [默认1]: ").strip()
        if sub == "2":
            for i, f in enumerate(files, start=1):
                print(f"{i}. {os.path.basename(f)}")
            try:
                idx = int(input("选择编号 [默认1]: ") or "1")
            except Exception:
                idx = 1
            idx = max(1, min(idx, len(files)))
            files = [files[idx - 1]]
        return PlaylistAudioSource(files, sample_rate=Config.SAMPLE_RATE)
    except Exception as e:
        print(f"音频测试选择流程出现异常：{e}，回退到麦克风。")
        return None



def _ensure_qt_plugins_env() -> None:
    """在 Windows 环境下设置 Qt 插件路径，避免找不到 platform plugin。"""
    if sys.platform.startswith("win"):
        # 优先尝试 PySide6 的插件路径（wheel 包内置完整 Qt）
        if not os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH"):
            try:
                import PySide6  # 延迟导入，仅用于定位路径
                base = os.path.dirname(PySide6.__file__)
                qpa = os.path.join(base, "plugins", "platforms")
                if os.path.isdir(qpa):
                    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = qpa
            except Exception:
                pass
        if not os.environ.get("QT_PLUGIN_PATH"):
            try:
                import PySide6
                base = os.path.dirname(PySide6.__file__)
                plugins = os.path.join(base, "plugins")
                if os.path.isdir(plugins):
                    os.environ["QT_PLUGIN_PATH"] = plugins
            except Exception:
                pass
        # 优先设置更精确的 QPA platform 路径
        if not os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH"):
            try:
                import PyQt5  # 延迟导入，仅用于定位路径
                base = os.path.dirname(PyQt5.__file__)
                qpa = os.path.join(base, "Qt", "plugins", "platforms")
                if os.path.isdir(qpa):
                    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = qpa
            except Exception:
                pass
        # 兼容性设置：部分环境需要设置更广义的插件目录
        if not os.environ.get("QT_PLUGIN_PATH"):
            try:
                import PyQt5
                base = os.path.dirname(PyQt5.__file__)
                plugins = os.path.join(base, "Qt", "plugins")
                if os.path.isdir(plugins):
                    os.environ["QT_PLUGIN_PATH"] = plugins
            except Exception:
                pass


if __name__ == "__main__":
    main()