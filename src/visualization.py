"""信号可视化模块（纯绘图，不依赖 UI）。"""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as scipy_signal


def plot_constellation(signal, title: str = "星座图", ax=None,
                       modulation_type: Optional[str] = None,
                       samples_per_symbol: int = 5, max_symbols: int = 500):
    """绘制星座图：提取信号的 I/Q 分量，在复平面上画散点并标出理想星座点。

    无论实信号（BPSK，Q 恒为 0）还是复信号（QPSK，I/Q 都有值），都统一画成
    复平面散点图，横轴 In-phase (I)、纵轴 Quadrature (Q)。

    ``samples_per_symbol`` 必须与 ``ModemSystem.bit_duration`` 一致：
    每个符号占这么多个样点，只取每个符号的第一个点，避免过采样点糊成一团。
    原来写死为 5，一旦改 bit_duration 星座图就会画错。
    """
    if ax is None:
        _fig, ax = plt.subplots(figsize=(6, 6))

    sig = np.asarray(signal)
    step = max(int(samples_per_symbol), 1)
    samples = sig[::step][:max_symbols]

    # 提取 I/Q 分量：BPSK 为实信号（Q 全 0），QPSK 为复信号
    inphase = np.real(samples)
    quadrature = np.imag(samples) if np.iscomplexobj(samples) else np.zeros_like(inphase)

    ax.scatter(inphase, quadrature, s=5, alpha=0.5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("In-phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.set_title(title)

    # 理想星座点
    ideal_i: List[float] = []
    ideal_q: List[float] = []
    if modulation_type == "BPSK":
        ideal_i = [-1, 1]
        ideal_q = [0, 0]
    elif modulation_type == "QPSK":
        ideal_points = [p / np.sqrt(2) for p in [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]]
        ideal_i = [float(p.real) for p in ideal_points]
        ideal_q = [float(p.imag) for p in ideal_points]

    if ideal_i:
        ax.plot(ideal_i, ideal_q, "ro", markersize=10, alpha=0.8, label="理想星座点")
        ax.legend()

    # 坐标轴范围：覆盖数据与理想点，并留边距，保证 ±1 的点不被裁掉
    x_values = np.concatenate([inphase, np.asarray(ideal_i)]) if ideal_i else inphase
    y_values = np.concatenate([quadrature, np.asarray(ideal_q)]) if ideal_q else quadrature
    xmax = max(np.max(np.abs(x_values)) if len(x_values) else 1.0, 1.0) + 0.5
    ymax = max(np.max(np.abs(y_values)) if len(y_values) else 1.0, 1.0) + 0.5
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-ymax, ymax)
    ax.set_aspect("equal", adjustable="box")

    return ax


def plot_spectrum(signal, title: str = "频谱图", ax=None, fs: int = 1000):
    """绘制幅度谱。

    实信号（BPSK）频谱共轭对称，只画正频率 0~fs/2 即可。
    复基带信号（QPSK）频谱左右不对称，必须 fftshift 后画完整的
    -fs/2~fs/2，只取前半会丢掉一半信息。

    注意：``fs`` 只是给横轴一个可读刻度的假定采样率。调制输出是
    「每比特 bit_duration 个样点」的基带序列，没有物理采样率，
    因此横轴数值只有相对意义。
    """
    if ax is None:
        _fig, ax = plt.subplots(figsize=(10, 4))

    sig = np.asarray(signal)
    if sig.size == 0:
        ax.set_title(f"{title}（无数据）")
        ax.grid(True, alpha=0.3)
        return ax

    fft_result = np.fft.fft(sig)
    freq = np.fft.fftfreq(sig.size, 1 / fs)

    if np.iscomplexobj(sig):
        freq_axis = np.fft.fftshift(freq)
        magnitude = np.abs(np.fft.fftshift(fft_result))
        xlim = (-fs / 2, fs / 2)
        color = None
    else:
        half = max(sig.size // 2, 1)
        freq_axis = freq[:half]
        magnitude = np.abs(fft_result[:half])
        xlim = (0, fs / 2)
        color = "green"

    ax.plot(freq_axis, 20 * np.log10(magnitude + 1e-10), linewidth=1, alpha=0.8, color=color)
    ax.set_xlabel(f"等效频率 (Hz，假定采样率 fs={fs})")
    ax.set_ylabel("幅度 (dB)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(xlim)
    return ax


def plot_spectrogram(signal, title: str = "时频图（Spectrogram）", ax=None, fs: int = 1000, nperseg: int = 256):
    """绘制时频图（Spectrogram）。"""
    if ax is None:
        _fig, ax = plt.subplots(figsize=(10, 6))

    f, t, sxx = scipy_signal.spectrogram(signal, fs=fs, nperseg=nperseg, mode="magnitude")
    im = ax.pcolormesh(t, f, 10 * np.log10(sxx + 1e-10), shading="gouraud", cmap="viridis")
    ax.set_ylabel("频率 (Hz)")
    ax.set_xlabel("时间 (s)")
    ax.set_title(title)
    # 传了 ax 就该用它所属 figure 的 colorbar，不依赖 pyplot 的"当前 figure"全局状态
    ax.figure.colorbar(im, ax=ax, label="幅度 (dB)")
    return ax
