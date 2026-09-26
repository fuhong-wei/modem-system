"""信号可视化模块（纯绘图，不依赖 UI）。"""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as scipy_signal


def plot_constellation(signal, title: str = "星座图", ax=None, modulation_type: Optional[str] = None):
    """绘制星座图：提取信号的 I/Q 分量，在复平面上画散点并标出理想星座点。

    无论实信号（BPSK，Q 恒为 0）还是复信号（QPSK，I/Q 都有值），都统一画成
    复平面散点图，横轴 In-phase (I)、纵轴 Quadrature (Q)。
    """
    if ax is None:
        _fig, ax = plt.subplots(figsize=(6, 6))

    samples = np.asarray(signal[:1000:5] if len(signal) > 1000 else signal)

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
    """绘制单边幅度谱。"""
    if ax is None:
        _fig, ax = plt.subplots(figsize=(10, 4))

    fft_result = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(signal), 1 / fs)
    positive_freq = freq[:len(freq) // 2]
    positive_fft = np.abs(fft_result[:len(fft_result) // 2])

    if np.iscomplexobj(signal):
        ax.plot(positive_freq, 20 * np.log10(positive_fft + 1e-10), linewidth=1, alpha=0.8)
    else:
        ax.plot(positive_freq, 20 * np.log10(positive_fft + 1e-10), linewidth=1, alpha=0.8, color="green")

    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("幅度 (dB)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, fs / 2])
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
    plt.colorbar(im, ax=ax, label="幅度 (dB)")
    return ax
