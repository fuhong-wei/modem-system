"""可靠 UDP 文件传输与调制解调系统 —— Streamlit 界面层。

核心逻辑位于 ``src/`` 包中，本文件只负责 UI 与交互。
"""

import glob
import os
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.transport import ReliableUDPTransfer
from src.visualization import plot_constellation, plot_spectrum, plot_spectrogram

# 设置 matplotlib 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

CODING_SCHEME_MAP = {
    "重复编码": "repetition",
    "汉明编码": "hamming",
    "无编码": "none",
}


# ---------------------------------------------------------------------------
# 文件管理辅助函数
# ---------------------------------------------------------------------------
def clear_received_files(save_dir: str) -> bool:
    """清除所有接收的文件。"""
    try:
        if not os.path.exists(save_dir):
            st.warning("保存目录不存在")
            return False

        deleted_count = 0
        total_size = 0
        for file_path in glob.glob(os.path.join(save_dir, "*")):
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)
                os.remove(file_path)
                deleted_count += 1

        st.success(f"✅ 已清除 {deleted_count} 个文件，释放 {total_size / 1024 / 1024:.2f} MB 空间")
        return True
    except Exception as e:
        st.error(f"清除文件时出错: {str(e)}")
        return False


def clear_temp_analysis_files() -> bool:
    """清除临时分析文件。"""
    try:
        if not os.path.exists("./temp_analysis"):
            return False

        deleted_count = 0
        for file_path in glob.glob("./temp_analysis/*.dat") + glob.glob("./temp_analysis/*.json"):
            if os.path.isfile(file_path):
                os.remove(file_path)
                deleted_count += 1

        st.success(f"✅ 已清除 {deleted_count} 个临时分析文件")
        return True
    except Exception as e:
        st.error(f"清除临时文件时出错: {str(e)}")
        return False


def get_folder_size(save_dir: str) -> int:
    """获取文件夹总大小（字节）。"""
    total_size = 0
    if os.path.exists(save_dir):
        for dirpath, _dirnames, filenames in os.walk(save_dir):
            for filename in filenames:
                total_size += os.path.getsize(os.path.join(dirpath, filename))
    return total_size


# ---------------------------------------------------------------------------
# 误码率分析（展示层）
# ---------------------------------------------------------------------------
def perform_ber_analysis(transfer: ReliableUDPTransfer) -> None:
    """执行并展示完整的误码率分析。"""
    try:
        if "sent_data" not in st.session_state:
            st.warning("❌ session state中缺少发送数据，尝试从文件加载...")
            sent_data = transfer.load_transmission_data_from_file("sent")
            if sent_data:
                st.session_state.sent_data = sent_data
                st.success("✅ 从文件成功加载发送数据")
            else:
                st.error("❌ 缺少发送数据，无法进行误码率分析")
                return

        sent_data = st.session_state.sent_data
        original_bits = sent_data["original_bits"]

        if "received_data" not in st.session_state:
            st.warning("⚠️ session state中缺少接收数据，尝试从文件加载...")
            received_data = transfer.load_transmission_data_from_file("received")
            # 只有「同一个文件、且不早于本次发送」的接收记录才允许参与对比，
            # 否则会拿旧文件的接收数据跟新文件的发送数据比，凭空得出约 50% 的假误码率
            sent_name = sent_data.get("filename")
            sent_ts = sent_data.get("timestamp") or 0
            if (
                received_data
                and received_data.get("filename") == sent_name
                and (received_data.get("timestamp") or 0) >= sent_ts - 1
            ):
                st.session_state.received_data = received_data
                st.success("✅ 从文件成功加载接收数据")
            elif received_data:
                st.warning(
                    f"⚠️ 磁盘上最新的接收记录是「{received_data.get('filename')}」，"
                    f"与本次发送的「{sent_name}」不匹配，已忽略；本次仅进行模拟信道分析"
                )
            else:
                st.warning("⚠️ 无法加载接收数据，将仅进行模拟信道分析")

        coding_scheme = CODING_SCHEME_MAP.get(sent_data["coding_scheme"], "repetition")
        modulation_type = sent_data["modulation_type"]
        snr_db = sent_data["snr_db"]

        st.info(f"开始误码率分析: 发送数据{len(original_bits)}比特")
        st.write(f"模拟参数: 调制={modulation_type}, 编码={coding_scheme}, 信噪比={snr_db}dB")

        # 1. 完整信道模拟（封顶 64KB，避免大文件 OOM；实际误码率用完整采样）
        simulated_max_bits = 64 * 1024 * 8
        sim_bits = original_bits[:simulated_max_bits]
        simulation_results = transfer.modem.simulate_complete_channel(
            sim_bits, modulation_type, coding_scheme, snr_db
        )
        # 把模拟用的原始比特放进结果，供可视化做「原始比特 vs 解码比特」的对比
        simulation_results["original_bits"] = sim_bits
        if len(original_bits) > simulated_max_bits:
            st.info(
                f"⚠️ 模拟信道分析封顶为前 64KB（{simulated_max_bits:,} 比特），"
                f"实际误码率仍对比完整 {len(original_bits):,} 比特。"
            )
        for step in simulation_results.get("steps", []):
            st.write(step)

        # 2. 实际传输误码率计算
        st.write("**实际传输结果:**")

        transfer_incomplete = False
        if "received_data" in st.session_state:
            received_data = st.session_state.received_data
            received_bits = received_data["received_bits"]
            completion_ratio = received_data.get("completion_ratio")

            # completion_ratio 为 None 表示来源不明（从没有元数据的旧 .dat 恢复），
            # 无法确认字节是否错位，和"确认不完整"同等处理，否则会假报警
            if completion_ratio is None or completion_ratio < 1.0:
                # 传输不完整时字节会错位，实际误码率无意义
                transfer_incomplete = True
                if completion_ratio is None:
                    st.write("- 传输完整度: 未知（接收记录缺少完整度信息）")
                    st.write("- 实际误码率: ⚠️ 无法确认传输是否完整，误码率不可信")
                else:
                    st.write(f"- 传输完整度: {completion_ratio:.1%}（不完整）")
                    st.write("- 实际误码率: ⚠️ 传输不完整，误码率无意义")
                actual_ber = None
                actual_errors = 0
                min_len_actual = 0
                data_consistent = False
                received_bits_sample = []
            else:
                # 一次比较同时得到错误数和误码率；原来先手算一遍再调 calculate_ber
                # 又比一遍，大文件下等于把全量对比做了两次
                min_len_actual = min(len(original_bits), len(received_bits))
                if min_len_actual > 0:
                    diff = np.asarray(original_bits[:min_len_actual]) != np.asarray(received_bits[:min_len_actual])
                    actual_errors = int(np.count_nonzero(diff))
                    actual_ber = actual_errors / min_len_actual
                else:
                    actual_errors = 0
                    actual_ber = 0.0
                data_consistent = actual_errors == 0

                st.write(f"- 对比比特数: {min_len_actual}")
                st.write(f"- 错误比特数: {actual_errors}")
                st.write(f"- 实际误码率: {actual_ber:.6f}")
                st.write(f"- 数据一致性: {'✅ 完全一致' if data_consistent else '❌ 不一致'}")
                received_bits_sample = received_bits[:100] if len(received_bits) > 100 else received_bits

            received_filename = received_data["filename"]
            st.write(
                f"- 接收端实际参数: 调制={received_data['modulation_type']}, "
                f"编码={received_data['coding_scheme']}, 信噪比={received_data['snr_db']}dB"
            )
        else:
            st.write("- 对比比特数: 无接收数据")
            st.write("- 错误比特数: N/A")
            st.write("- 实际误码率: N/A")
            st.write("- 数据一致性: N/A")

            actual_ber = None
            actual_errors = 0
            min_len_actual = 0
            data_consistent = False
            received_filename = "无接收数据"
            received_bits_sample = []

        # 3. 保存分析结果
        recv_record = st.session_state.get("received_data") or {}
        st.session_state.ber_analysis_results = {
            "actual_ber": actual_ber,
            "actual_errors": actual_errors,
            "actual_total_bits": min_len_actual,
            "data_consistent": data_consistent,
            "transfer_incomplete": transfer_incomplete,
            "simulated_ber": simulation_results["simulated_ber"],
            "simulated_errors": simulation_results["errors"],
            "simulated_total_bits": simulation_results["compared_bits"],
            "modulation_type": modulation_type,
            "coding_scheme": sent_data["coding_scheme"],
            "snr_db": snr_db,
            "sent_filename": sent_data["filename"],
            "received_filename": received_filename,
            "file_size": sent_data["file_size"],
            "analyzed_bytes": sent_data.get("analyzed_bytes"),
            "duration": sent_data.get("duration"),
            "speed": sent_data.get("speed"),
            "retransmissions": sent_data.get("retransmissions"),
            "retransmitted_chunks": sent_data.get("retransmitted_chunks"),
            "total_chunks": sent_data.get("total_chunks"),
            "recv_completion_ratio": recv_record.get("completion_ratio"),
            "recv_missing_chunks": recv_record.get("missing_chunks"),
            "coding_rate": transfer.modem.coding_rate,
            "original_bits_count": len(original_bits),
            "encoded_bits_count": len(simulation_results["encoded_bits"]),
            "signal_power": simulation_results["signal_power"],
            "noise_power": simulation_results["noise_power"],
            "actual_snr": simulation_results["actual_snr"],
            "simulated_bits_sample": simulation_results["decoded_bits"][:100],
            "original_bits_sample": original_bits[:100],
            "received_bits_sample": received_bits_sample,
            "simulation_results": simulation_results,
        }

        if "ber_analysis_needed" in st.session_state:
            del st.session_state.ber_analysis_needed

        if actual_ber is not None:
            st.success(
                f"✅ 误码率分析完成! 实际传输BER: {actual_ber:.6f}, "
                f"模拟信道BER: {simulation_results['simulated_ber']:.6f}"
            )
        else:
            st.success(f"✅ 误码率分析完成! 模拟信道BER: {simulation_results['simulated_ber']:.6f} (无接收数据)")

        st.markdown("---")
        display_signal_visualization_enhanced(transfer, simulation_results, modulation_type)

    except Exception as e:
        import traceback

        st.error(f"误码率分析错误: {str(e)}")
        st.error(f"详细错误信息: {traceback.format_exc()}")


# ---------------------------------------------------------------------------
# 信号可视化（展示层）
# ---------------------------------------------------------------------------
def display_signal_visualization_enhanced(
    transfer: ReliableUDPTransfer, simulation_results: dict, modulation_type: str
) -> None:
    """显示增强版信号可视化：星座图、频谱、综合视图。"""
    try:
        st.subheader("📈 高级信号可视化")

        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["🔵 星座图分析", "📊 频谱分析", "🌊 综合视图"])

        with viz_tab1:
            st.info("星座图展示了调制信号在复平面上的分布，直观显示信号的相位和幅度信息")

            if modulation_type == "BPSK":
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

                sps = transfer.modem.bit_duration
                plot_constellation(
                    simulation_results["modulated_signal"][:500],
                    title="BPSK调制信号星座图", ax=ax1, modulation_type=modulation_type,
                    samples_per_symbol=sps,
                )
                plot_constellation(
                    simulation_results["noisy_signal"][:500],
                    title="BPSK加噪信号星座图", ax=ax2, modulation_type=modulation_type,
                    samples_per_symbol=sps,
                )

                display_length = min(100, len(simulation_results["modulated_signal"]))
                time_axis = np.arange(display_length)
                ax3.plot(time_axis, np.real(simulation_results["modulated_signal"][:display_length]),
                         "b-", label="调制信号", linewidth=1.5)
                ax3.plot(time_axis, np.real(simulation_results["noisy_signal"][:display_length]),
                         "r-", alpha=0.6, label="加噪信号", linewidth=1)
                ax3.set_title("BPSK时域信号对比")
                ax3.set_xlabel("时间")
                ax3.set_ylabel("幅度")
                ax3.grid(True, alpha=0.3)
                ax3.legend()

                if len(simulation_results["modulated_signal"]) > 0:
                    mod_amplitude = np.real(simulation_results["modulated_signal"][:500])
                    noisy_amplitude = np.real(simulation_results["noisy_signal"][:500])
                    ax4.hist(mod_amplitude, bins=30, alpha=0.5, label="调制信号幅度", color="blue")
                    ax4.hist(noisy_amplitude, bins=30, alpha=0.5, label="加噪信号幅度", color="red")
                    ax4.set_xlabel("幅度")
                    ax4.set_ylabel("频数")
                    ax4.set_title("幅度分布对比")
                    ax4.legend()
                    ax4.grid(True, alpha=0.3)

            elif modulation_type == "QPSK":
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))

                sps = transfer.modem.bit_duration
                # 取 1000 个样点（而非 200），除以 bit_duration 后才有足够的符号点画出星座簇
                plot_constellation(
                    simulation_results["modulated_signal"][:1000],
                    title="QPSK调制信号星座图", ax=ax1, modulation_type=modulation_type,
                    samples_per_symbol=sps,
                )
                plot_constellation(
                    simulation_results["noisy_signal"][:1000],
                    title="QPSK加噪信号星座图", ax=ax2, modulation_type=modulation_type,
                    samples_per_symbol=sps,
                )

                if len(simulation_results["modulated_signal"]) > 0:
                    mod_phase = np.angle(simulation_results["modulated_signal"][:500])
                    noisy_phase = np.angle(simulation_results["noisy_signal"][:500])
                    ax3.hist(mod_phase, bins=30, alpha=0.5, label="调制信号相位", color="blue", density=True)
                    ax3.hist(noisy_phase, bins=30, alpha=0.5, label="加噪信号相位", color="red", density=True)
                    ax3.set_xlabel("相位 (弧度)")
                    ax3.set_ylabel("概率密度")
                    ax3.set_title("相位分布对比")
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)

                    mod_amplitude = np.abs(simulation_results["modulated_signal"][:500])
                    noisy_amplitude = np.abs(simulation_results["noisy_signal"][:500])
                    ax4.hist(mod_amplitude, bins=30, alpha=0.5, label="调制信号幅度", color="blue", density=True)
                    ax4.hist(noisy_amplitude, bins=30, alpha=0.5, label="加噪信号幅度", color="red", density=True)
                    ax4.set_xlabel("幅度")
                    ax4.set_ylabel("概率密度")
                    ax4.set_title("幅度分布对比")
                    ax4.legend()
                    ax4.grid(True, alpha=0.3)

            else:
                # 兜底分支：避免新增调制方式时 fig 未定义导致 UnboundLocalError
                fig, ax_na = plt.subplots(figsize=(6, 4))
                ax_na.text(0.5, 0.5, f"暂不支持的调制方式: {modulation_type}",
                           ha="center", va="center", transform=ax_na.transAxes)
                ax_na.set_axis_off()

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.info("""
            **星座图分析说明:**
            - 🔵 **蓝色点**: 调制信号在复平面上的位置
            - 🔴 **红色点**: 理想星座点的位置
            - 📊 **分布宽度**: 表示信号的噪声水平，分布越宽噪声越大
            - 🔄 **相位偏移**: 显示信号是否受到相位噪声影响
            - 📏 **幅度变化**: 显示信号是否受到幅度衰落影响
            """)

        with viz_tab2:
            st.info("频谱图展示了信号在不同频率上的能量分布，反映信号的频率特性")

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

            fs = 1000
            plot_spectrum(simulation_results["modulated_signal"][:2000], title="调制信号频谱", ax=ax1, fs=fs)
            plot_spectrum(simulation_results["noisy_signal"][:2000], title="加噪信号频谱", ax=ax2, fs=fs)

            # FFT 长度必须动态取，不能写死 2000：
            # 信号短于 1000 点时 x/y 长度不等会直接抛异常，
            # 1000~2000 点之间虽然不崩，但 fftfreq 按 2000 算出来的横轴数值是错的
            n_fft = min(
                2000,
                len(simulation_results["modulated_signal"]),
                len(simulation_results["noisy_signal"]),
            )
            fft_mod = np.fft.fft(simulation_results["modulated_signal"][:n_fft])
            fft_noisy = np.fft.fft(simulation_results["noisy_signal"][:n_fft])
            freq = np.fft.fftfreq(n_fft, 1 / fs)

            half = max(n_fft // 2, 1)
            positive_freq = freq[:half]
            positive_fft_mod = np.abs(fft_mod[:half])
            positive_fft_noisy = np.abs(fft_noisy[:half])

            ax3.plot(positive_freq, 20 * np.log10(positive_fft_mod + 1e-10),
                     "b-", alpha=0.7, label="原始信号", linewidth=1)
            ax3.plot(positive_freq, 20 * np.log10(positive_fft_noisy + 1e-10),
                     "r-", alpha=0.5, label="加噪信号", linewidth=1)
            ax3.set_xlabel("频率 (Hz)")
            ax3.set_ylabel("幅度 (dB)")
            ax3.set_title("频谱对比")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim([0, fs / 2])

            # 噪声谱要用复频谱相减再取模：两个模值相减不等于噪声的模（噪声有相位），
            # 而且结果可能为负，取 abs 后波形完全失真
            noise_estimate = np.abs(fft_noisy[:half] - fft_mod[:half])
            ax4.plot(positive_freq, 20 * np.log10(noise_estimate + 1e-10),
                     "g-", alpha=0.7, label="估计噪声", linewidth=1)
            ax4.set_xlabel("频率 (Hz)")
            ax4.set_ylabel("噪声幅度 (dB)")
            ax4.set_title("噪声频谱估计")
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_xlim([0, fs / 2])

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.info("""
            **频谱分析说明:**
            - 📶 **主瓣**: 信号的主要能量集中在主瓣频率
            - 📉 **旁瓣**: 主瓣之外的频率分量，反映信号的频谱泄露
            - 🎚️ **带宽**: 信号占据的频率范围
            - 📊 **噪声基底**: 频谱图中的平坦部分表示噪声水平
            - 🔄 **频谱形状**: 不同调制方式有不同的频谱特征
            """)

        with viz_tab3:
            st.info("综合视图提供信号的时频联合分析，展示信号随时间变化的频率特性")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            try:
                plot_spectrogram(
                    simulation_results["modulated_signal"][:4000],
                    title="调制信号时频图（Spectrogram）", ax=ax1, fs=1000, nperseg=256,
                )
            except Exception as e:
                ax1.text(0.5, 0.5, f"时频图生成失败: {str(e)}",
                         ha="center", va="center", transform=ax1.transAxes)
                ax1.set_title("调制信号时频图（Spectrogram）")

            try:
                plot_spectrogram(
                    simulation_results["noisy_signal"][:4000],
                    title="加噪信号时频图（Spectrogram）", ax=ax2, fs=1000, nperseg=256,
                )
            except Exception as e:
                ax2.text(0.5, 0.5, f"时频图生成失败: {str(e)}",
                         ha="center", va="center", transform=ax2.transAxes)
                ax2.set_title("加噪信号时频图（Spectrogram）")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("⏱️ 时域波形对比")
            fig2, (ax3, ax4, ax5) = plt.subplots(3, 1, figsize=(14, 10))

            display_length = min(200, len(simulation_results["modulated_signal"]))
            time_axis = np.arange(display_length)

            mod_signal = simulation_results["modulated_signal"][:display_length]
            noisy_signal = simulation_results["noisy_signal"][:display_length]

            ax3.plot(time_axis, np.real(mod_signal),
                     "b-", label="原始信号(实部)", linewidth=1.5, alpha=0.8)
            ax3.plot(time_axis, np.real(noisy_signal),
                     "r-", label="加噪信号(实部)", linewidth=1, alpha=0.6)
            ax3.set_xlabel("时间")
            ax3.set_ylabel("幅度")
            ax3.set_title("信号实部对比")
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            if np.iscomplexobj(mod_signal):
                ax4.plot(time_axis, np.imag(mod_signal),
                         "g-", label="原始信号(虚部)", linewidth=1.5, alpha=0.8)
                ax4.plot(time_axis, np.imag(noisy_signal),
                         "orange", label="加噪信号(虚部)", linewidth=1, alpha=0.6)
                ax4.set_xlabel("时间")
                ax4.set_ylabel("幅度")
                ax4.set_title("信号虚部对比")
            else:
                ax4.plot(time_axis, np.abs(mod_signal),
                         color="purple", linestyle="-", label="原始信号幅度", linewidth=1.5, alpha=0.8)
                ax4.plot(time_axis, np.abs(noisy_signal),
                         color="brown", linestyle="-", label="加噪信号幅度", linewidth=1, alpha=0.6)
                ax4.set_xlabel("时间")
                ax4.set_ylabel("幅度")
                ax4.set_title("信号幅度对比")
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            # 噪声分量：直接画出加噪与原始的差值，让差异一目了然
            noise_signal = noisy_signal - mod_signal
            ax5.plot(time_axis, np.real(noise_signal),
                     color="red", label="噪声分量(实部)", linewidth=1, alpha=0.7)
            if np.iscomplexobj(noise_signal):
                ax5.plot(time_axis, np.imag(noise_signal),
                         color="orange", label="噪声分量(虚部)", linewidth=1, alpha=0.7)
            ax5.set_xlabel("时间")
            ax5.set_ylabel("幅度")
            ax5.set_title("噪声分量（加噪信号 − 原始信号）")
            ax5.legend()
            ax5.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

            st.info("""
            **综合视图分析说明:**
            - 🌊 **时频图（Spectrogram）**: 展示信号频率随时间的变化，反映信号的时频特性
            - ⏱️ **时域波形**: 展示信号幅度随时间的变化
            - 🔄 **实部/虚部**: 对于复信号，分别展示同相分量和正交分量
            - 📈 **信号变化**: 观察信号在传输过程中的畸变和失真
            - 🎨 **颜色映射**: 时频图中颜色深浅表示信号强度
            """)

        # 比特错误可视化
        st.subheader("🔍 比特错误分析")

        orig_all = np.asarray(simulation_results["original_bits"])
        deco_all = np.asarray(simulation_results["decoded_bits"])
        n_cmp = min(len(orig_all), len(deco_all))
        orig_all, deco_all = orig_all[:n_cmp], deco_all[:n_cmp]

        # 先在全量比特里找错误，再把观察窗对准第一个错误。
        # 窗口死钉在开头时，高信噪比下前 50 比特几乎不会出错，
        # 用户永远看到"无错误"，误以为对比功能没生效。
        all_errors = np.flatnonzero(orig_all != deco_all)
        window = min(50, n_cmp)
        start = 0
        if all_errors.size and n_cmp > window:
            start = max(0, min(int(all_errors[0]) - window // 4, n_cmp - window))

        display_bits = window
        original_display = orig_all[start:start + window]
        simulated_display = deco_all[start:start + window]

        if all_errors.size:
            st.caption(
                f"全部 {n_cmp:,} 个对比比特中共有 {all_errors.size:,} 个错误"
                f"（误码率 {all_errors.size / n_cmp:.6f}）；"
                f"下图窗口已对准第一个错误（第 {int(all_errors[0]):,} 比特）"
            )
        else:
            st.caption(f"全部 {n_cmp:,} 个对比比特中无错误；下图显示开头 {window} 比特")

        fig3, ax = plt.subplots(figsize=(15, 4))
        x_pos = np.arange(start, start + window)

        error_positions = []
        for i in range(window):
            if original_display[i] != simulated_display[i]:
                error_positions.append(start + i)
                ax.axvspan(start + i - 0.4, start + i + 0.4, alpha=0.3, color="red")

        ax.stem(x_pos, original_display, linefmt="b-", markerfmt="bo", basefmt=" ", label="原始比特")
        ax.stem(x_pos + 0.1, simulated_display, linefmt="r-", markerfmt="rx", basefmt=" ", label="模拟接收比特")

        ax.set_xlabel("比特位置")
        ax.set_ylabel("比特值")
        ax.set_title(
            f"比特对比：第 {start:,}~{start + window - 1:,} 比特"
            f"（红色区域为错误，窗口内 {len(error_positions)} 个 / 全量 {all_errors.size:,} 个）"
        )
        ax.set_ylim(-0.5, 1.5)
        ax.legend()
        ax.grid(True, alpha=0.3)

        st.pyplot(fig3)
        plt.close(fig3)

        if error_positions:
            st.write(f"**错误比特位置 (第 {start:,}~{start + window - 1:,} 比特窗口内):** {error_positions}")
        else:
            st.write(f"**错误比特位置:** 第 {start:,}~{start + window - 1:,} 比特窗口内无错误")

    except Exception as e:
        import traceback

        st.warning(f"增强信号可视化显示失败: {e}")
        st.error(f"详细错误: {traceback.format_exc()}")


# ---------------------------------------------------------------------------
# 接收端实时面板
# ---------------------------------------------------------------------------
@st.fragment(run_every="1s")
def _render_receiver_panel(transfer: ReliableUDPTransfer, monitor_quality: bool) -> None:
    """接收端实时面板：监听状态、端口切换提示、错误、进度与日志（每秒自动刷新）。"""
    events = transfer.drain_receiver_events()

    # 累积消息到 session_state，避免刷新后丢失
    if "receiver_log" not in st.session_state:
        st.session_state.receiver_log = []
    st.session_state.receiver_log.extend(events["messages"])
    st.session_state.receiver_log = st.session_state.receiver_log[-50:]

    # 监听状态
    if transfer.receiver_alive:
        port_note = f"（端口 {transfer.receiver_actual_port}）" if transfer.receiver_actual_port else ""
        st.info(f"🔴 接收端正在监听中...{port_note}")
    else:
        st.info("⚪ 接收端未运行")

    if transfer.receiver_error:
        st.error(transfer.receiver_error)

    done, total = events["progress"]
    if total > 0:
        st.progress(min(done / total, 1.0))
    if events["status"]:
        st.caption(events["status"])

    # 接收日志
    if st.session_state.receiver_log:
        with st.expander(f"📜 接收日志（{len(st.session_state.receiver_log)} 条）", expanded=True):
            for level, text in st.session_state.receiver_log:
                show = {"success": st.success, "error": st.error, "warning": st.warning, "info": st.info}.get(
                    level, st.info
                )
                show(text)

    # 信道质量（真实统计，来自最近一次接收记录）
    if transfer.receiver_alive and monitor_quality:
        st.subheader("📡 实际信道质量（最近一次传输）")
        received = st.session_state.get("received_data") or {}
        total_chunks = received.get("total_chunks")
        if total_chunks:
            completion = received.get("completion_ratio") or 0
            missing = received.get("missing_chunks") or 0
            recv_loss = missing / total_chunks * 100
            speed = received.get("speed")
            duration = received.get("duration")
            col_qual1, col_qual2, col_qual3 = st.columns(3)
            with col_qual1:
                st.metric("接收端完整度", f"{completion:.1%}")
            with col_qual2:
                st.metric("接收端丢包率", f"{recv_loss:.2f}%")
            with col_qual3:
                st.metric("平均速度", f"{speed:.2f} MB/s" if speed is not None else "N/A")
            if duration is not None:
                st.caption(f"传输耗时 {duration:.2f} 秒，共 {total_chunks} 块，缺失 {missing} 块")
        else:
            st.info("等待首次传输完成… 接收成功后这里会显示真实的完整度 / 丢包率 / 速度。")


# ---------------------------------------------------------------------------
# 主界面
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="可靠UDP文件传输系统", page_icon="📡", layout="wide")

    st.title("📡 可靠UDP文件传输与调制解调系统")
    st.markdown("基于UDP协议的可靠文件传输，集成调制解调、编码解码和误码率分析功能")

    # 初始化传输器
    if "transfer" not in st.session_state:
        def _on_message(level: str, text: str) -> None:
            {"success": st.success, "error": st.error, "warning": st.warning, "info": st.info}.get(
                level, st.info
            )(text)

        st.session_state.transfer = ReliableUDPTransfer(on_message=_on_message)

    transfer: ReliableUDPTransfer = st.session_state.transfer

    # 初始化 session state
    if "coding_scheme" not in st.session_state:
        st.session_state.coding_scheme = "重复编码"
    if "ber_analysis_results" not in st.session_state:
        st.session_state.ber_analysis_results = None
    if "ber_analysis_needed" not in st.session_state:
        st.session_state.ber_analysis_needed = False

    # 系统参数配置
    st.sidebar.header("⚙️ 通信系统参数")

    modulation_type = st.sidebar.selectbox(
        "调制方式", ["BPSK", "QPSK"], index=0,
        help="BPSK: 抗噪性好，1比特/符号\nQPSK: 频谱效率高，2比特/符号",
    )

    coding_scheme = st.sidebar.selectbox(
        "编码方案", ["重复编码", "汉明编码", "无编码"], index=0,
        help="重复编码: (1,3)简单可靠\n汉明编码: (7,4)效率较高\n无编码: 原始传输",
    )
    st.session_state.coding_scheme = coding_scheme

    snr_db = st.sidebar.slider(
        "信道信噪比 (dB)", min_value=-30, max_value=30, value=10,
        help="模拟实际信道噪声条件，值越小噪声越大",
    )

    st.sidebar.header("🚀 传输协议参数")

    transfer.chunk_size = st.sidebar.selectbox(
        "数据块大小", [1024, 2048, 4096, 8192], index=2,
        help="较大的数据块提高传输效率，但增加丢包风险",
    )

    transfer.window_size = st.sidebar.slider(
        "滑动窗口大小", min_value=1, max_value=16, value=8,
        help="控制同时传输的数据包数量，影响吞吐量",
    )

    transfer.max_retries = st.sidebar.slider(
        "最大重传次数", min_value=1, max_value=10, value=5,
        help="数据包丢失时的重传次数，影响可靠性",
    )

    transfer.timeout = st.sidebar.slider(
        "超时时间(秒)", min_value=1, max_value=10, value=3,
        help="等待确认的超时时间，影响响应性",
    )

    sample_size_label = st.sidebar.selectbox(
        "分析采样大小", ["1KB", "64KB", "1MB", "整个文件"], index=1,
        help="误码率分析取多少字节数据；越大越能区分不同文件，但模拟信道分析会封顶 64KB",
    )
    transfer.sample_size = {
        "1KB": 1024, "64KB": 64 * 1024, "1MB": 1024 * 1024, "整个文件": None,
    }[sample_size_label]

    if coding_scheme == "重复编码":
        transfer.modem.coding_rate = 1 / 3
    elif coding_scheme == "汉明编码":
        transfer.modem.coding_rate = 4 / 7
    else:
        transfer.modem.coding_rate = 1

    # 把侧边栏的选择同步到 modem，使「从 .dat 恢复且无元数据」时的兜底值
    # 至少跟用户当前的界面设置一致，而不是永远的 BPSK / 10dB
    transfer.modem.modulation_type = modulation_type
    transfer.modem.snr_db = snr_db

    tab1, tab2, tab3 = st.tabs(["📤 发送文件", "📥 接收文件", "📊 误码率分析"])

    # ---------------- 发送文件 ----------------
    with tab1:
        st.header("文件发送端")
        st.info("""
        **可靠传输特性:**
        - 🎯 滑动窗口协议提高吞吐量
        - 🔄 自动重传丢失的数据包
        - 📈 实时速度和进度监控
        - ⚡ 连接超时检测和恢复
        """)

        col1, col2 = st.columns(2)

        with col1:
            target_ip = st.text_input("目标IP地址", "127.0.0.1")
            target_port = st.number_input("目标端口", min_value=1000, max_value=65535, value=8888)

        with col2:
            uploaded_file = st.file_uploader(
                "选择要发送的文件", type=["txt", "jpg", "png", "pdf", "zip", "mp3", "mp4"]
            )

        if uploaded_file is not None:
            file_size = len(uploaded_file.getvalue())
            file_info = {
                "文件名": uploaded_file.name,
                "文件大小": f"{file_size / 1024 / 1024:.2f} MB" if file_size > 1024 * 1024 else f"{file_size / 1024:.2f} KB",
                "调制方式": modulation_type,
                "编码方案": coding_scheme,
                "编码速率": f"{transfer.modem.coding_rate:.3f}",
                "信噪比": f"{snr_db} dB",
                "数据块大小": f"{transfer.chunk_size} bytes",
                "窗口大小": transfer.window_size,
            }
            st.json(file_info)

            estimated_chunks = (file_size + transfer.chunk_size - 1) // transfer.chunk_size
            estimated_time = estimated_chunks * 0.01
            if estimated_time > 1:
                st.info(f"预计传输时间: {estimated_time:.1f} 秒")

        if st.button("🚀 开始发送", type="primary"):
            if uploaded_file is None:
                st.warning("请先选择要发送的文件")
            elif target_ip.strip() in ("127.0.0.1", "localhost", "::1") and not transfer.receiver_alive:
                st.error(
                    "❌ 目标为本地主机，但接收端未在监听。"
                    "请先切换到「接收文件」页签点「▶️ 开始监听」，再回来发送。"
                )
            else:
                with st.spinner("建立连接并发送文件中..."):
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()

                    def _on_progress(done: int, total: int) -> None:
                        progress_bar.progress(min(done / total, 1.0) if total else 0.0)

                    def _on_status(text: str) -> None:
                        status_text.text(text)

                    transfer.on_progress = _on_progress
                    transfer.on_status = _on_status

                    record = transfer.send_file(
                        target_ip, target_port, uploaded_file.getvalue(), uploaded_file.name,
                        modulation_type, coding_scheme, snr_db,
                    )
                    if record:
                        st.session_state.sent_data = record
                        # 清空旧接收数据与分析结果，避免「新发送 vs 旧接收」跨文件对比
                        st.session_state.pop("received_data", None)
                        st.session_state.ber_analysis_results = None
                        st.balloons()
                        st.success("✅ 文件发送成功完成!")
                    else:
                        st.error("❌ 文件发送失败")
        elif uploaded_file is None:
            st.warning("请先选择要发送的文件")

    # ---------------- 接收文件 ----------------
    with tab2:
        st.header("文件接收端")
        st.info("""
        **接收端特性:**
        - ✅ 自动确认接收的数据包
        - 🔄 处理重复和乱序数据包
        - 📊 实时进度和速度显示
        - 🛡️ 不完整文件恢复机制
        - 🗑️ 文件管理功能
        - 📡 根据发送端参数自动调整解调解码
        """)

        col1, col2 = st.columns(2)

        with col1:
            listen_port = st.number_input("监听端口", min_value=1000, max_value=65535, value=8888)
            save_dir = st.text_input("保存目录", "./received_files")

            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                st.success(f"创建目录: {save_dir}")

        with col2:
            monitor_quality = st.checkbox(
                "监控实际信道质量", value=True,
                help="启用后会在接收端显示实际信道质量统计",
            )

            if st.button("▶️ 开始监听", type="primary") and not transfer.receiver_alive:
                transfer.stop_receiver()  # 清理可能残留的旧 socket / 停止事件

                def receiver_thread() -> None:
                    record = transfer.start_receiver(listen_port, save_dir)
                    if record:
                        transfer.set_last_received_record(record)

                thread = threading.Thread(target=receiver_thread)
                thread.daemon = True
                thread.start()
                st.success("接收端已启动! 等待连接...")

            if st.button("⏹️ 停止监听") and transfer.receiver_alive:
                transfer.stop_receiver()
                st.success("接收端已停止")

        # 接收线程产出结果 -> 主线程写入 session_state（后台线程不直接写 session_state）
        record = transfer.take_last_received_record()
        if record is not None:
            st.session_state.received_data = record
            st.session_state.ber_analysis_needed = True

        # 接收端实时面板：状态 / 端口切换 / 错误 / 进度 / 日志（每秒自动刷新）
        _render_receiver_panel(transfer, monitor_quality)

        # 文件管理
        st.subheader("🗂️ 文件管理")

        if os.path.exists(save_dir):
            folder_size_mb = get_folder_size(save_dir) / 1024 / 1024
            received_files = [f for f in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, f))]
            file_count = len(received_files)

            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("文件数量", file_count)
            with col_info2:
                st.metric("文件夹大小", f"{folder_size_mb:.2f} MB")
            with col_info3:
                if st.button("🗑️ 清除所有文件", type="secondary"):
                    if clear_received_files(save_dir):
                        st.rerun()

        if os.path.exists(save_dir):
            received_files = [f for f in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, f))]
            if received_files:
                st.subheader("📁 已接收文件")

                received_files.sort(key=lambda f: os.path.getmtime(os.path.join(save_dir, f)), reverse=True)

                files_per_page = 10
                total_files = len(received_files)
                total_pages = (total_files + files_per_page - 1) // files_per_page

                if "current_page" not in st.session_state:
                    st.session_state.current_page = 1

                col_page1, col_page2, col_page3 = st.columns([1, 2, 1])
                with col_page1:
                    if st.button("⬅️ 上一页") and st.session_state.current_page > 1:
                        st.session_state.current_page -= 1
                        st.rerun()
                with col_page2:
                    st.write(f"第 {st.session_state.current_page} 页 / 共 {total_pages} 页")
                with col_page3:
                    if st.button("下一页 ➡️") and st.session_state.current_page < total_pages:
                        st.session_state.current_page += 1
                        st.rerun()

                start_idx = (st.session_state.current_page - 1) * files_per_page
                end_idx = min(start_idx + files_per_page, total_files)
                current_files = received_files[start_idx:end_idx]

                for file in current_files:
                    file_path = os.path.join(save_dir, file)
                    file_size = os.path.getsize(file_path)
                    file_mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(file_path)))

                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        size_str = (
                            f"{file_size / 1024 / 1024:.2f} MB"
                            if file_size > 1024 * 1024
                            else f"{file_size / 1024:.2f} KB"
                        )
                        st.text(f"{file} ({size_str})")
                        st.caption(f"接收时间: {file_mtime}")
                    with col2:
                        with open(file_path, "rb") as f:
                            st.download_button("📥 下载", f, file_name=file, key=f"dl_{file}")
                    with col3:
                        if st.button("🗑️", key=f"del_{file}"):
                            try:
                                os.remove(file_path)
                                st.success(f"已删除: {file}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"删除失败: {str(e)}")

                st.info(f"显示 {start_idx + 1}-{end_idx} 个文件，共 {total_files} 个文件")

    # ---------------- 误码率分析 ----------------
    with tab3:
        st.header("📊 误码率分析")

        if st.session_state.ber_analysis_needed and (
            "sent_data" in st.session_state or "received_data" in st.session_state
        ):
            st.info("🔄 检测到新的传输数据，正在自动进行误码率分析...")
            perform_ber_analysis(transfer)

        col_manual1, col_manual2, col_manual3 = st.columns([1, 1, 1])
        with col_manual1:
            if st.button("🔄 手动进行误码率分析", type="primary"):
                data_available = False
                if "sent_data" in st.session_state:
                    data_available = True
                else:
                    sent_data = transfer.load_transmission_data_from_file("sent")
                    if sent_data:
                        st.session_state.sent_data = sent_data
                        data_available = True
                        st.success("✅ 从文件成功加载发送数据")

                if data_available:
                    if "ber_analysis_results" in st.session_state:
                        del st.session_state.ber_analysis_results
                    with st.spinner("正在进行误码率分析..."):
                        perform_ber_analysis(transfer)
                else:
                    st.error("❌ 缺少发送数据，无法进行分析。请先发送一个文件。")

        with col_manual2:
            if st.button("🗑️ 清除分析结果"):
                if "ber_analysis_results" in st.session_state:
                    del st.session_state.ber_analysis_results
                if "ber_analysis_needed" in st.session_state:
                    del st.session_state.ber_analysis_needed
                st.success("分析结果已清除")
                st.rerun()

        with col_manual3:
            if st.button("🧹 清除临时文件"):
                if clear_temp_analysis_files():
                    st.rerun()

        if "ber_analysis_results" in st.session_state and st.session_state.ber_analysis_results is not None:
            results = st.session_state.ber_analysis_results

            st.success("✅ 误码率分析结果")

            st.subheader("📋 传输基本信息")
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.write(f"**发送文件**: {results['sent_filename']}")
                st.write(f"**接收文件**: {results['received_filename']}")
                st.write(f"**文件大小**: {results['file_size'] / 1024:.2f} KB")
                st.write(f"**原始比特数**: {results['original_bits_count']:,}")
            with col_info2:
                st.write(f"**调制方式**: {results['modulation_type']}")
                st.write(f"**编码方案**: {results['coding_scheme']}")
                st.write(f"**编码效率**: {results['coding_rate']:.3f}")
                st.write(f"**设定信噪比**: {results['snr_db']} dB")
                _snr = results.get("actual_snr")
                _snr_ok = isinstance(_snr, (int, float)) and np.isfinite(_snr)
                st.write(f"**实际信噪比**: {f'{_snr:.2f} dB' if _snr_ok else 'N/A'}")

            st.subheader("⚡ 本次传输性能")
            analyzed_bytes = results.get("analyzed_bytes")
            analyzed_bits = results.get("original_bits_count")
            duration = results.get("duration")
            speed = results.get("speed")
            retransmissions = results.get("retransmissions")
            retransmitted_chunks = results.get("retransmitted_chunks")
            total_chunks = results.get("total_chunks")
            recv_completion = results.get("recv_completion_ratio")
            recv_missing = results.get("recv_missing_chunks")
            # 丢包率 = 发生过重传的「块数」/ 总块数（同一块重传多次只算一次丢包）
            sent_loss = (retransmitted_chunks / total_chunks * 100) if (retransmitted_chunks is not None and total_chunks) else None
            # 重传率 = 总重传「次数」/ 总块数，同一块重传多次会累计，可能超过 100%
            retrans_rate = (retransmissions / total_chunks * 100) if (retransmissions is not None and total_chunks) else None
            recv_loss = (recv_missing / total_chunks * 100) if (recv_missing is not None and total_chunks) else None

            col_perf1, col_perf2, col_perf3 = st.columns(3)
            with col_perf1:
                st.metric("分析范围", f"{analyzed_bytes / 1024:.1f} KB" if analyzed_bytes else "N/A",
                          help=f"实际误码率对比 {analyzed_bits:,} 比特；模拟信道分析封顶 64KB")
                st.metric("传输耗时", f"{duration:.2f} 秒" if duration is not None else "N/A")
            with col_perf2:
                st.metric("平均速度", f"{speed:.2f} MB/s" if speed is not None else "N/A")
                st.metric("重传次数", f"{retransmissions}" if retransmissions is not None else "N/A")
            with col_perf3:
                st.metric("发送端丢包率", f"{sent_loss:.2f}%" if sent_loss is not None else "N/A",
                          help="发生过重传的块数 / 总块数")
                st.metric("发送端重传率", f"{retrans_rate:.2f}%" if retrans_rate is not None else "N/A",
                          help="总重传次数 / 总块数；同一块重传多次会累计，可能超过 100%")
                st.metric("接收端丢包率", f"{recv_loss:.2f}%" if recv_loss is not None else "N/A",
                          help="缺失块数 / 总块数")
            if recv_completion is not None:
                if total_chunks:
                    st.caption(
                        f"接收端完整度：{recv_completion:.1%}"
                        f"（收到 {int(recv_completion * total_chunks)}/{total_chunks} 块）"
                    )
                else:
                    st.caption(f"接收端完整度：{recv_completion:.1%}（总块数未知）")

            st.subheader("🎯 误码率对比分析")
            col_ber1, col_ber2 = st.columns(2)

            with col_ber1:
                if results.get("transfer_incomplete"):
                    st.metric("实际传输误码率", "传输不完整", help="传输不完整，字节错位，误码率无意义")
                    st.warning("**⚠️ 传输不完整**: 数据块缺失导致字节错位，实际误码率无意义")
                elif results["actual_ber"] is not None:
                    actual_ber_display = f"{results['actual_ber']:.8f}" if results["actual_ber"] > 0 else "0.000000"
                    delta_text = f"{results['actual_errors']} 错误比特" if results["actual_errors"] > 0 else "0 错误比特"
                    st.metric("实际传输误码率", actual_ber_display, delta=delta_text,
                              help="实际传输过程中的误码率，对比发送和接收数据计算得出")
                    if results["actual_ber"] == 0:
                        st.success("""
                        **✅ 可靠传输机制验证:**
                        - 滑动窗口协议确保数据有序传输
                        - 自动重传机制纠正丢失的数据包
                        - 确认机制保证数据完整性
                        - 因此实际传输误码率为0
                        """)
                    else:
                        st.warning("""
                        **⚠️ 传输过程中检测到错误:**
                        - 可能存在网络问题
                        - 数据包可能损坏
                        - 建议检查网络连接
                        """)
                else:
                    st.metric("实际传输误码率", "N/A", delta="无接收数据",
                              help="没有接收数据可用于计算实际传输误码率")
                    st.info("**ℹ️ 无接收数据**: 无法计算实际传输误码率")

            with col_ber2:
                st.metric(
                    "模拟信道误码率", f"{results['simulated_ber']:.8f}",
                    delta=f"{results['simulated_errors']} 错误比特",
                    help=f"在 {results['snr_db']}dB 信噪比下对发送数据模拟完整信道传输的误码率",
                )
                st.info("""
                **模拟信道分析:**
                - 发送数据 → 编码 → 调制 → 加噪 → 解调 → 解码
                - 模拟完整无线信道传输过程
                - 展示系统在实际信道中的抗噪性能
                """)

            st.subheader("📊 详细统计")
            col_stat1, col_stat2 = st.columns(2)

            with col_stat1:
                st.write("**🔵 实际传输统计**")
                if results["actual_ber"] is not None:
                    st.write(f"- 对比比特数: {results['actual_total_bits']:,}")
                    st.write(f"- 错误比特数: {results['actual_errors']:,}")
                    st.write(f"- 误码率: {results['actual_ber']:.8f}")
                    st.write(f"- 正确率: {(1 - results['actual_ber']) * 100:.6f}%")
                    st.write(f"- 数据一致性: {'✅ 完全一致' if results['data_consistent'] else '❌ 不一致'}")
                else:
                    st.write("- 对比比特数: N/A")
                    st.write("- 错误比特数: N/A")
                    st.write("- 误码率: N/A")
                    st.write("- 正确率: N/A")
                    st.write("- 数据一致性: N/A")

            with col_stat2:
                st.write("**🔴 模拟信道统计**")
                st.write(f"- 对比比特数: {results['simulated_total_bits']:,}")
                st.write(f"- 错误比特数: {results['simulated_errors']:,}")
                st.write(f"- 误码率: {results['simulated_ber']:.8f}")
                st.write(f"- 正确率: {(1 - results['simulated_ber']) * 100:.6f}%")

            st.subheader("🔋 功率和信噪比信息")
            col_power1, col_power2, col_power3 = st.columns(3)
            with col_power1:
                st.write(f"**信号功率**: {results.get('signal_power', 0):.6f}")
                st.write(f"**噪声功率**: {results.get('noise_power', 0):.6f}")
            actual_snr = results.get("actual_snr")
            actual_snr_ok = isinstance(actual_snr, (int, float)) and np.isfinite(actual_snr)
            with col_power2:
                st.write(f"**设定信噪比**: {results['snr_db']} dB")
                st.write(f"**实际信噪比**: {f'{actual_snr:.2f} dB' if actual_snr_ok else 'N/A'}")
            with col_power3:
                if actual_snr_ok:
                    snr_error = actual_snr - results["snr_db"]
                    st.write(f"**信噪比误差**: {snr_error:.2f} dB")
                    if abs(snr_error) < 0.5:
                        st.success("✅ 信噪比控制精确")
                    elif abs(snr_error) < 1.0:
                        st.info("ℹ️ 信噪比控制良好")
                    else:
                        st.warning("⚠️ 信噪比控制有偏差")
                else:
                    st.write("**信噪比误差**: N/A")
                    st.info("ℹ️ 实际信噪比不可用（噪声功率为 0 或数据缺失）")

            st.subheader("🔍 数据样本对比")
            col_sample1, col_sample2, col_sample3 = st.columns(3)
            with col_sample1:
                st.write("**原始发送数据 (前100比特)**")
                st.text("".join(map(str, results.get("original_bits_sample", []))))
            with col_sample2:
                st.write("**模拟信道数据 (前100比特)**")
                st.text("".join(map(str, results.get("simulated_bits_sample", []))))
            with col_sample3:
                if len(results.get("received_bits_sample", [])) > 0:
                    st.write("**实际接收数据 (前100比特)**")
                    st.text("".join(map(str, results.get("received_bits_sample", []))))
                else:
                    st.write("**实际接收数据**")
                    st.info("无接收数据")

            if "original_bits_sample" in results and "simulated_bits_sample" in results:
                error_positions = []
                min_sample_len = min(len(results["original_bits_sample"]), len(results["simulated_bits_sample"]))
                for i in range(min_sample_len):
                    if results["original_bits_sample"][i] != results["simulated_bits_sample"][i]:
                        error_positions.append(i)
                if error_positions:
                    st.write(f"**模拟信道错误位置 (前100比特中):** {error_positions}")
                else:
                    st.write("**模拟信道错误位置:** 前100比特中无错误")

            st.subheader("📈 系统性能评估")
            col_perf1, col_perf2, col_perf3 = st.columns(3)
            with col_perf1:
                if results["actual_ber"] is not None:
                    if results["actual_ber"] == 0:
                        st.success("✅ 实际传输: 完美无误 (可靠传输机制)")
                    elif results["actual_ber"] < 0.001:
                        st.info("ℹ️ 实际传输: 良好")
                    else:
                        st.error("🔴 实际传输: 存在错误")
                else:
                    st.info("ℹ️ 实际传输: 无数据")
            with col_perf2:
                if results["simulated_ber"] < 0.001:
                    st.success("✅ 模拟信道: 优秀抗噪")
                elif results["simulated_ber"] < 0.01:
                    st.info("ℹ️ 模拟信道: 良好抗噪")
                else:
                    st.error("🔴 模拟信道: 抗噪较差")
            with col_perf3:
                if results["actual_ber"] is not None:
                    if results["actual_ber"] == 0 and results["simulated_ber"] > 0:
                        st.success("🎯 可靠性增益: 无限 (可靠传输机制)")
                    elif results["actual_ber"] < results["simulated_ber"]:
                        improvement_factor = (
                            results["simulated_ber"] / results["actual_ber"] if results["actual_ber"] > 0 else float("inf")
                        )
                        st.success(f"🎯 可靠性增益: {improvement_factor:.2f}倍")
                    else:
                        st.info("ℹ️ 可靠性增益: 无明显提升")
                else:
                    st.info("ℹ️ 可靠性增益: 无法计算")

            st.subheader("📊 误码率可视化")
            fig, ax = plt.subplots(figsize=(10, 6))

            if results["actual_ber"] is not None:
                categories = ["实际传输", "模拟信道"]
                ber_values = [results["actual_ber"], results["simulated_ber"]]
                colors = ["#4CAF50", "#FF6B6B"]
            else:
                categories = ["模拟信道"]
                ber_values = [results["simulated_ber"]]
                colors = ["#FF6B6B"]

            # 对数坐标画不出 0（log(0) = -inf），给 0 值一个可见的地板高度，
            # 否则"实际传输 BER = 0"这根柱子会整根消失，看着像图没画出来
            use_log = max(ber_values) > 0
            positive = [v for v in ber_values if v > 0]
            floor = (min(positive) / 100) if (use_log and positive) else 1e-10
            plot_values = [(v if v > 0 else floor) for v in ber_values] if use_log else ber_values

            bars = ax.bar(categories, plot_values, color=colors, alpha=0.7, edgecolor="black")
            for bar, value in zip(bars, ber_values):
                display_text = "0（无误码）" if value == 0 else f"{value:.8f}"
                ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                        display_text, ha="center", va="bottom", fontweight="bold")

            ax.set_ylabel("误码率 (BER)")
            ax.set_title("实际传输 vs 模拟信道误码率对比")
            ax.grid(True, alpha=0.3)

            if use_log:
                ax.set_yscale("log")
                ax.set_ylim(bottom=floor / 10)
                ax.set_ylabel("误码率 (BER) - 对数坐标，0 值以图示最低高度表示")

            st.pyplot(fig)
            plt.close(fig)

            st.subheader("💡 系统优化建议")
            if results["simulated_ber"] > 0.01:
                st.error("""
                **建议改进措施:**
                - 提高信噪比设置
                - 使用更强大的编码方案（如汉明编码）
                - 考虑使用QPSK调制提高抗噪性
                - 增加重复编码的重复次数
                """)
            elif results["simulated_ber"] > 0.001:
                st.warning("""
                **可选的改进:**
                - 适度提高信噪比
                - 尝试不同的编码方案
                - 验证当前配置是否满足需求
                """)
            else:
                st.success("""
                **当前配置良好:**
                - 系统在模拟信道中表现优秀
                - 当前参数设置合理
                - 可考虑优化传输效率
                """)

            st.markdown("---")
            if st.button("🔄 重新显示高级信号可视化", key="redisplay_viz"):
                if "simulation_results" in results:
                    display_signal_visualization_enhanced(transfer, results["simulation_results"], results["modulation_type"])
                else:
                    st.warning("无法重新显示可视化，缺少模拟结果数据")

        else:
            st.warning("🔍 暂无误码率分析数据")
            st.info("""
            ### 如何获取误码率分析数据：

            1. **配置通信参数** - 在左侧边栏设置调制方式、编码方案和信噪比(-30dB到30dB)
            2. **发送文件** - 在"发送文件"标签页选择文件并发送
            3. **接收文件** - 在"接收文件"标签页启动接收端接收文件（可选）
            4. **查看分析** - 完成传输后返回此页面查看误码率分析

            ### 数据持久化特性：

            **数据备份机制:**
            - 📁 **文件备份**: 所有传输数据都会备份到临时文件
            - 🔄 **自动恢复**: 即使页面刷新或会话中断，数据也能从文件恢复
            - 🛡️ **容错处理**: 即使文件接收不完整，也会保存已接收部分用于分析
            - 🧹 **清理功能**: 可以手动清除临时文件释放空间

            ### 关于实际传输误码率的说明：

            **实际传输误码率计算方式:**
            - 🔄 **比特对比**: 对比发送和接收的数据（范围由侧边栏「分析采样大小」决定，默认前 64KB）
            - ✅ **错误检测**: 统计不一致的比特数量
            - 📊 **误码率计算**: 错误比特数 / 总对比比特数
            - 🛡️ **可靠传输**: 由于重传机制，通常误码率为0

            ### 模拟信道误码率的意义：

            模拟信道误码率展示了系统在**真实无线环境**中的性能（不考虑重传机制）:
            - 📡 模拟完整的通信链路：编码 → 调制 → 加噪 → 解调 → 解码
            - 🌪️ 考虑实际信道噪声和干扰的影响
            - 🔧 评估不同调制编码方案的抗噪性能
            - 📊 为系统优化提供理论依据
            - ⚠️ 展示的是原始信道性能，没有重传机制保护

            **对比意义**:
            - 🟢 **实际传输BER**: 展示可靠传输机制的有效性
            - 🔴 **模拟信道BER**: 展示原始信道条件下的系统抗噪性能
            - 📈 **差值**: 体现重传机制对系统可靠性的提升

            **信噪比范围说明**:
            - -30dB 到 -10dB: 极低信噪比，误码率很高
            - -10dB 到 0dB: 低信噪比，有明显误码
            - 0dB 到 10dB: 中等信噪比，误码率较低
            - 10dB 到 20dB: 高信噪比，误码率很低
            - 20dB 到 30dB: 极高信噪比，几乎无误码
            """)

            st.subheader("当前系统配置")
            col_conf1, col_conf2 = st.columns(2)
            with col_conf1:
                st.write(f"**调制方式**: {modulation_type}")
                st.write(f"**编码方案**: {coding_scheme}")
                st.write(f"**编码速率**: {transfer.modem.coding_rate:.3f}")
            with col_conf2:
                st.write(f"**信噪比**: {snr_db} dB")
                st.write(f"**数据块大小**: {transfer.chunk_size} bytes")
                st.write(f"**窗口大小**: {transfer.window_size}")


if __name__ == "__main__":
    main()
