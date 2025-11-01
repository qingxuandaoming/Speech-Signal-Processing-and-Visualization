# 实时语音信号处理系统（real_time_voice_processing）

## 项目概述

本项目是一个基于Python的实时语音信号处理系统，实现了音频信号的实时采集、处理和可视化。系统采用多线程架构设计，能够实时计算语音信号的短时能量、过零率等特征参数，并进行语音活动检测。

## 技术特点

- **实时音频采集**：使用PyAudio库实现跨平台的音频输入
- **多线程架构**：采用生产者-消费者模式，确保实时性
- **信号处理算法**：实现短时能量、过零率、自相关等核心算法
- **实时可视化**：使用PyQtGraph实现动态图表显示
- **语音活动检测**：基于能量和过零率的双门限检测算法

## 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   音频采集线程   │    │   信号处理线程   │    │   可视化线程    │
│  (Producer)     │───>│  (Consumer)     │───>│  (Consumer)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 功能模块

### 1. 音频采集模块
- 支持16kHz采样率的音频输入
- 实时音频流处理
- 多平台兼容性

### 2. 信号处理模块
- **短时能量计算**：反映信号的能量变化
- **过零率计算**：区分清音和浊音
- **自相关分析**：用于基音周期检测
- **语音活动检测**：识别语音片段

### 3. 可视化模块
- 实时波形显示
- 能量变化趋势图
- 过零率动态显示
- 语音检测结果指示

## 安装说明

### 环境要求
- Python 3.10（推荐并已配置）
- PyAudio 0.2.14
- PyQtGraph 0.13.3
- Qt 绑定（PyQt5 5.15.x 或 PySide6）
- NumPy 1.26.4
- SciPy 1.12.0

### 安装步骤

```powershell
# Windows（PowerShell）创建并激活虚拟环境（Python 3.10）
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 安装依赖（使用 requirements.txt）
pip install -r real_time_voice_processing\requirements.txt

# 可选：使用 pyproject.toml（需要较新 pip）
pip install .

# 也可使用脚本自动创建虚拟环境
pwsh ./scripts/setup_venv.ps1
```

```bash
# Linux/macOS（bash/zsh）
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 注意事项

**Windows系统**：

- PyAudio在Windows上可能需要预编译轮子；建议使用`py -3.10`创建环境。
- 若安装失败，可使用conda：`conda install pyaudio` 或通过第三方镜像安装。

## PyAudio 安装 FAQ（Windows）

- 问：`pip install pyaudio` 报错，提示缺少 `portaudio.h` 或编译失败？
  - 答：Windows 下推荐使用预编译轮子（whl）。步骤：
    1) 访问可信的预编译轮子源，下载与你的 Python 版本与架构匹配的文件，例如 `PyAudio‑0.2.14‑cp310‑cp310‑win_amd64.whl`。
    2) 在虚拟环境中执行：`pip install <下载的whl文件路径>`。

- 问：如何确认我的 Python/架构版本？
  - 答：使用 `python -V` 查看版本；使用 `python -c "import platform; print(platform.architecture())"` 查看架构（`64bit`/`32bit`）。

- 问：仍安装失败怎么办？
  - 答：可尝试：
    - 使用 `conda install pyaudio`（如果你使用的是 Anaconda/Miniconda）。
    - 安装 Visual C++ Build Tools 后再编译安装，但不推荐此路径给初学者。

- 问：如何验证安装成功？
  - 答：在虚拟环境中执行：`python -c "import pyaudio; print(pyaudio.__version__)"`，若无异常且输出版本号则安装成功。

**Linux系统**：

- 需要安装PortAudio开发包：`sudo apt-get install portaudio19-dev`

## 使用方法

### 基本使用

```bash
# 运行主程序
python main.py

# 运行单元测试
pytest

# 运行演示脚本
python real_time_voice_processing/demo.py
```

### 功能说明

1. **开始处理**：点击"开始处理"按钮启动音频采集和处理
2. **停止处理**：点击"停止处理"按钮停止系统运行
3. **保存数据**：点击"保存数据"按钮保存处理结果到文件

### 配置参数

可以在`config.py`文件中调整以下参数：

- **音频参数**：采样率、帧长、帧移等
- **算法参数**：能量阈值、过零率阈值等
- **界面参数**：更新频率、显示范围等

## 算法原理

### 短时能量

短时能量反映了语音信号在短时间内的能量变化，计算公式为：

$$
E(i) = \sum_{n=0}^{N-1} x_i(n)^2
$$

其中 $x_i(n)$ 为第 $i$ 帧加窗后的语音信号，$N$ 为帧长。

### 过零率

过零率表示单位时间内信号穿过零轴的次数，计算公式为：

$$
\mathrm{ZCR}(i) = \frac{1}{2N} \sum_{n=0}^{N-1} \big|\operatorname{sign}(x_i(n)) - \operatorname{sign}(x_i(n+1))\big|
$$

其中 $\operatorname{sign}(\cdot)$ 是符号函数，$N$ 为帧长。

### 预加重

为提升高频分量、抑制直流漂移，使用预加重滤波：

$$
y[n] = x[n] - \alpha\, x[n-1], \quad \alpha \in (0,1) \text{，默认 } \alpha=0.97
$$

### 语音活动检测

采用双门限检测算法：
1. 第一级：能量检测，过滤掉低能量信号
2. 第二级：过零率检测，进一步确认语音信号

可形式化表示为：

$$
\mathrm{VAD}(i) = \begin{cases}
1, & E(i) > T_E \;\land\; \mathrm{ZCR}(i) > T_Z \\
0, & \text{otherwise}
\end{cases}
$$

## 性能优化

### 实时性保证

1. **多线程设计**：将采集、处理、显示分离到不同线程
2. **缓冲区管理**：合理设置缓冲区大小平衡延迟和稳定性
3. **算法优化**：使用NumPy向量化计算提高效率

### 内存管理

1. **循环缓冲区**：使用deque实现固定大小的循环缓冲区
2. **数据清理**：及时清理不再需要的数据
3. **内存映射**：大数据量处理时使用内存映射文件

## 测试验证

使用 `pytest` 运行测试用例：

```bash
pytest
```

测试包括：
- 窗函数生成测试
- 短时能量计算测试
- 过零率计算测试
- 自相关函数测试
- 语音活动检测测试

## 扩展功能

### 潜在扩展方向

1. **基音周期检测**：基于自相关函数实现基音检测
2. **噪声抑制**：添加自适应滤波算法
3. **特征提取**：实现MFCC等频域特征提取
4. **语音识别**：集成第三方语音识别API

### 性能优化

1. **GPU加速**：使用CUDA加速信号处理算法
2. **算法优化**：实现更高效的信号处理算法
3. **硬件优化**：针对特定硬件平台进行优化

## 项目结构

```
real_time_voice_processing/
├── main.py                  # 入口文件（调用 runtime 与 ui）
├── signal_processing/       # 信号处理算法子包（模块化实现）
│   ├── __init__.py          # 聚合类 SignalProcessing（兼容旧 API）
│   ├── windows.py           # 窗函数
│   ├── preprocessing.py     # 预加重与分帧
│   ├── time_features.py     # 时域特征
│   ├── frequency_features.py# 频域特征（Mel、MFCC、谱熵）
│   └── vad.py               # 固定/自适应 VAD
├── config.py                # 配置文件
├── runtime/
│   └── engine.py           # 运行时音频采集与处理（线程控制）
├── ui/
│   └── visualization.py    # 可视化界面与交互控件
├── tests/
│   └── test_signal_processing.py  # 基于 pytest 的算法单元测试
├── docs/
│   ├── 开发指南.md                # 开发与贡献指南
│   ├── 架构说明.md                # 系统架构与数据流程
│   ├── 算法说明.md                # 公式与算法细节（LaTeX）
│   └── 常见问题.md                # FAQ 与故障排除
├── README.md                # 项目说明文档
└── requirements.txt         # 依赖包列表
```

## 文档构建（Sphinx）

- 本项目使用 Sphinx 自动生成文档并集成现有 Markdown（MyST）。
- 构建命令（仓库根目录）：

```bash
python -m pip install -r real_time_voice_processing/requirements.txt
sphinx-build -b html docs docs/_build/html
```

- 打开 `docs/_build/html/index.html` 浏览完整说明与 API 文档。

## 参考文献

1. 语音信号处理（第三版），胡航等著
2. Digital Speech Processing，Lawrence Rabiner著
3. 实时信号处理系统设计，王金龙等著
4. Python数字信号处理，José Unpingco著

## 版本信息

- **版本**：v1.0
- **日期**：2025年10月21日
- **作者**：豆包编程助手

## 许可证

本项目采用MIT许可证，详情请参考LICENSE文件。