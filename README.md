# 可靠 UDP 文件传输与调制解调系统

![CI](https://github.com/fuhong-wei/modem-system/actions/workflows/ci.yml/badge.svg)

基于 [Streamlit](https://streamlit.io/) 的交互式通信系统，实现了基于 UDP 的可靠文件传输、数字调制解调（BPSK/QPSK）、信道编码（重复编码 / 汉明编码）与误码率（BER）分析。

## 技术栈

| 类别 | 技术 |
| --- | --- |
| 语言 | Python 3.10+ |
| 界面 | Streamlit |
| 科学计算 | NumPy、SciPy、Matplotlib |
| 测试 / 规范 | pytest、pytest-cov、ruff、mypy |
| CI/CD | GitHub Actions |

## 功能特性

- **调制解调**：BPSK / QPSK 调制与解调
- **信道编码**：重复编码 (1,3)、汉明编码 (7,4，支持单比特纠错)
- **可靠 UDP 传输**：数据分块、滑动窗口、超时重传、ACK 确认，在不可靠 UDP 之上实现可靠传输
- **误码率分析**：模拟信道与实际传输的误码率对比、比特错误定位
- **信号可视化**：星座图、频谱分析、时频瀑布图、时域波形对比
- **中文界面**：发送文件 / 接收文件 / 误码率分析 三个页签

## 系统架构

项目采用分层架构，核心逻辑与界面层分离，核心算法独立可测：

```mermaid
graph TB
    subgraph UI["展示层"]
        APP[app.py<br/>Streamlit 界面]
    end
    subgraph CORE["核心层 src/"]
        TR[transport.py<br/>可靠 UDP 传输]
        MD[modulation.py<br/>调制解调 / 信道模拟]
        CD[coding.py<br/>信道编码]
        VZ[visualization.py<br/>信号可视化]
    end
    APP --> TR
    APP --> VZ
    TR --> MD
    MD --> CD
```

- **`app.py`**：Streamlit 界面层，只负责交互与展示。
- **`src/transport.py`**：可靠 UDP 传输，通过回调接口与上层解耦。
- **`src/modulation.py`**：调制解调、加噪、误码率、信道模拟（纯逻辑）。
- **`src/coding.py`**：重复编码与汉明编码（纯函数）。
- **`src/visualization.py`**：matplotlib 绘图。

## 项目结构

```
.
├── app.py                    # Streamlit 入口（界面层）
├── src/                      # 核心库
│   ├── coding.py             #   信道编码
│   ├── modulation.py         #   调制解调与信道模拟
│   ├── transport.py          #   可靠 UDP 传输
│   └── visualization.py      #   信号可视化
├── tests/                    # 单元测试
├── .github/workflows/ci.yml  # GitHub Actions CI
├── pyproject.toml            # ruff / mypy / pytest 配置
├── requirements.txt          # 运行时依赖
├── requirements-dev.txt      # 开发依赖
└── README.md
```

## 环境要求

- Python 3.10+

## 安装

建议使用虚拟环境：

```bash
# 1. 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. 安装运行时依赖
pip install -r requirements.txt

# 3.（可选）安装开发依赖（测试与代码检查）
pip install -r requirements-dev.txt
```

## 使用方法

```bash
streamlit run app.py
```

**传输流程**（发送端与接收端可分别运行；演示时请先在接收端监听）：

1. **接收文件**：在接收端页签点「开始监听」（默认端口 8888）
2. **发送文件**：在发送端设置目标 IP 与端口，选择文件并点「开始发送」
3. **误码率分析**：切换页签，调整调制方式、编码方案与信噪比，观察误码率与信号波形

## 运行截图

> 待补充。运行 `streamlit run app.py` 后，可将界面截图放入 `docs/screenshots/` 目录，然后在下方用 `![说明](docs/screenshots/xxx.png)` 引用。

<!--
![发送文件](docs/screenshots/send.png)
![误码率分析](docs/screenshots/ber.png)
-->

## 测试与代码质量

```bash
# 运行单元测试并生成覆盖率报告
pytest --cov=src --cov-report=term-missing

# 静态检查（lint）
ruff check .

# 类型检查
mypy src
```

核心算法（编码 94%、调制解调 95%）已覆盖单元测试；CI 会在每次 push / PR 时自动运行上述全部检查。

## 主要模块

- `ModemSystem`：调制解调与信道模拟核心
- `ReliableUDPTransfer`：基于 UDP 的可靠传输（滑动窗口 + 重传）

## 开发记录

- [项目问题复盘与面试准备](PROBLEMS_AND_SOLUTIONS.md)：开发过程中踩过的坑、排查思路与解决方案（含面试问答准备）
- [接收端状态同步 Bug 复盘](PROJECT_REVIEW.md)：`receiver_alive` 竞态问题的详细复盘
