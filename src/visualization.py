"""信号可视化模块（纯绘图，不依赖 UI）。"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as scipy_signal


def plot_constellation(signal, title: str = "星座图", ax=None, modulation_type: str = None):
    """绘制星座图。"""
    if ax is None:
        _fig, ax = plt.subplots(figsize=(6, 6))

    samples = signal[:1000:5] if len(signal) > 1000 else signal

    if np.iscomplexobj(samples):
        ax.scatter(np.real(samples), np.imag(samples), alpha=0.6, s=20, edgecolors="b", linewidths=0.5)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("同相分量 (I)")
        ax.set_ylabel("正交分量 (Q)")
    else:
        ax.scatter(range(len(samples)), samples, alpha=0.6, s=20, edgecolors="b", linewidths=0.5)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("采样点")
        ax.set_ylabel("幅度")

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")

    if modulation_type == "BPSK":
        ax.plot([-1, 1], [0, 0], "ro", markersize=8, alpha=0.5, label="理想点")
        ax.legend()
    elif modulation_type == "QPSK":
        ideal_points = [p / np.sqrt(2) for p in [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]]
        ax.plot([p.real for p in ideal_points], [p.imag for p in ideal_points],
                "ro", markersize=8, alpha=0.5, label="理想点")
        ax.legend()

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


def plot_spectrogram(signal, title: str = "频谱瀑布图", ax=None, fs: int = 1000, nperseg: int = 256):
    """绘制频谱瀑布图（时频分析）。"""
    if ax is None:
        _fig, ax = plt.subplots(figsize=(10, 6))

    f, t, sxx = scipy_signal.spectrogram(signal, fs=fs, nperseg=nperseg, mode="magnitude")
    im = ax.pcolormesh(t, f, 10 * np.log10(sxx + 1e-10), shading="gouraud", cmap="viridis")
    ax.set_ylabel("频率 (Hz)")
    ax.set_xlabel("时间 (s)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="幅度 (dB)")
    return ax
