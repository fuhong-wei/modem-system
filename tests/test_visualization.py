"""visualization.py 的回归测试：只验证不崩 + 关键行为，不比对像素。"""

import matplotlib

matplotlib.use("Agg")  # 无头环境

import numpy as np
import pytest

from src.visualization import plot_constellation, plot_spectrogram, plot_spectrum


@pytest.mark.parametrize("n", [1, 7, 480, 1200, 2000])
def test_plot_spectrum_any_length_real(n):
    """实信号任意长度都不该崩（历史上短信号会因长度不匹配抛异常）。"""
    ax = plot_spectrum(np.random.randn(n), fs=1000)
    assert ax.get_xlim() == (0.0, 500.0)


def test_plot_spectrum_complex_is_two_sided():
    """复信号必须画双边谱 -fs/2~fs/2，只画正半轴会丢一半信息。"""
    sig = np.random.randn(512) + 1j * np.random.randn(512)
    ax = plot_spectrum(sig, fs=1000)
    assert ax.get_xlim() == (-500.0, 500.0)


def test_plot_spectrum_empty_does_not_raise():
    ax = plot_spectrum(np.array([]))
    assert "无数据" in ax.get_title()


def test_plot_constellation_bpsk_is_scatter_not_waveform():
    """BPSK 必须画成复平面散点（Q 全 0），而不是时域波形。"""
    sig = np.repeat(np.array([1, -1, 1, -1] * 50), 5).astype(float)
    ax = plot_constellation(sig, modulation_type="BPSK", samples_per_symbol=5)
    assert ax.get_xlabel() == "In-phase (I)"
    assert ax.get_ylabel() == "Quadrature (Q)"
    assert len(ax.collections) >= 1  # 有散点


def test_plot_constellation_respects_samples_per_symbol():
    """步长必须跟 samples_per_symbol 走，不能写死 5。"""
    sig = np.arange(80, dtype=float)
    ax = plot_constellation(sig, samples_per_symbol=8, max_symbols=500)
    pts = ax.collections[0].get_offsets()
    assert len(pts) == 10  # 80 / 8
    assert pts[1][0] == pytest.approx(8.0)


def test_plot_spectrogram_title_is_time_frequency():
    """时频图标题不能写成"频谱图"。"""
    ax = plot_spectrogram(np.random.randn(2000), fs=1000, nperseg=256)
    assert "时频" in ax.get_title()
    assert ax.get_xlabel() == "时间 (s)"
