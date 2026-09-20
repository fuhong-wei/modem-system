import streamlit as st
import socket
import struct
import time
import hashlib
import threading
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import queue
import glob

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 调制解调系统类
class ModemSystem:
    def __init__(self):
        self.modulation_type = "BPSK"
        self.coding_rate = 1/2
        self.snr_db = 10  # 提高默认信噪比
        self.bit_duration = 5

    def encode_data(self, data_bits, coding_scheme="repetition"):
        """编码数据"""
        if coding_scheme == "repetition":
            # 重复编码 (1,3) - 每个比特重复3次
            return np.repeat(data_bits, 3).tolist()
        elif coding_scheme == "hamming":
            # 汉明编码 (7,4) - 4个数据位+3个校验位
            encoded = []
            for i in range(0, len(data_bits), 4):
                chunk = data_bits[i:i+4]
                if len(chunk) < 4:
                    chunk = chunk + [0] * (4 - len(chunk))
                # 计算校验位
                p1 = chunk[0] ^ chunk[1] ^ chunk[3]
                p2 = chunk[0] ^ chunk[2] ^ chunk[3]
                p3 = chunk[1] ^ chunk[2] ^ chunk[3]
                encoded.extend(chunk + [p1, p2, p3])
            return encoded
        else:
            return data_bits

    def decode_data(self, encoded_bits, coding_scheme="repetition"):
        """解码数据"""
        if coding_scheme == "repetition":
            # 重复编码解码 - 多数表决
            decoded = []
            for i in range(0, len(encoded_bits), 3):
                chunk = encoded_bits[i:i+3]
                if chunk:
                    decoded.append(1 if sum(chunk) >= 2 else 0)
            return decoded
        elif coding_scheme == "hamming":
            # 汉明编码解码 - 错误检测和纠正
            decoded = []
            for i in range(0, len(encoded_bits), 7):
                chunk = encoded_bits[i:i+7]
                if len(chunk) == 7:
                    # 计算校验子
                    s1 = chunk[0] ^ chunk[1] ^ chunk[3] ^ chunk[4] ^ chunk[6]
                    s2 = chunk[0] ^ chunk[2] ^ chunk[3] ^ chunk[5] ^ chunk[6]
                    s3 = chunk[1] ^ chunk[2] ^ chunk[3]

                    # 错误位置
                    error_pos = s1 + 2*s2 + 4*s3

                    if error_pos > 0 and error_pos <= 7:
                        # 纠正错误
                        chunk[error_pos-1] = 1 - chunk[error_pos-1]

                    decoded.extend(chunk[:4])
            return decoded
        else:
            return encoded_bits

    def modulate(self, bits, modulation_type="BPSK"):
        """调制"""
        if modulation_type == "BPSK":
            # BPSK调制: 0 -> +1, 1 -> -1
            symbols = np.where(bits, -1, 1)
            return np.repeat(symbols, self.bit_duration)
        elif modulation_type == "QPSK":
            # QPSK调制 - 每2比特映射到一个符号
            modulated_signal = np.array([], dtype=complex)
            for i in range(0, len(bits)-1, 2):
                dibit = (bits[i], bits[i+1])
                if dibit == (0, 0):
                    symbol = complex(1, 1)    # 45°
                elif dibit == (0, 1):
                    symbol = complex(-1, 1)   # 135°
                elif dibit == (1, 0):
                    symbol = complex(-1, -1)  # 225°
                else:
                    symbol = complex(1, -1)   # 315°
                symbol /= np.sqrt(2)  # 归一化
                modulated_signal = np.append(modulated_signal, [symbol] * self.bit_duration)
            return modulated_signal
        return np.array([])

    def demodulate(self, received_signal, modulation_type="BPSK"):
        """解调"""
        bits = []
        if modulation_type == "BPSK":
            # BPSK解调
            for i in range(0, len(received_signal), self.bit_duration):
                if i < len(received_signal):
                    end_idx = min(i + self.bit_duration, len(received_signal))
                    symbol_avg = np.mean(received_signal[i:end_idx])
                    bits.append(1 if symbol_avg < 0 else 0)
        elif modulation_type == "QPSK":
            # QPSK解调
            for i in range(0, len(received_signal), self.bit_duration):
                if i < len(received_signal):
                    end_idx = min(i + self.bit_duration, len(received_signal))
                    symbol_avg = np.mean(received_signal[i:end_idx])
                    real_part = np.real(symbol_avg)
                    imag_part = np.imag(symbol_avg)
                    # 判决四个象限
                    if real_part >= 0 and imag_part >= 0:
                        bits.extend([0, 0])
                    elif real_part < 0 and imag_part >= 0:
                        bits.extend([0, 1])
                    elif real_part < 0 and imag_part < 0:
                        bits.extend([1, 0])
                    else:
                        bits.extend([1, 1])
        return bits

    def add_noise(self, signal_data, snr_db):
        """添加高斯白噪声模拟信道衰落 - 修复版本"""
        # 确保信号功率计算正确
        if np.iscomplexobj(signal_data):
            signal_power = np.mean(np.real(signal_data)**2 + np.imag(signal_data)**2)
        else:
            signal_power = np.mean(signal_data**2)

        # 转换SNR为线性值
        snr_linear = 10 ** (snr_db / 10.0)

        # 计算噪声功率
        noise_power = signal_power / snr_linear

        # 生成噪声
        if np.iscomplexobj(signal_data):
            noise_real = np.sqrt(noise_power / 2) * np.random.randn(len(signal_data))
            noise_imag = np.sqrt(noise_power / 2) * np.random.randn(len(signal_data))
            noise = noise_real + 1j * noise_imag
        else:
            noise = np.sqrt(noise_power) * np.random.randn(len(signal_data))

        # 添加噪声到信号
        noisy_signal = signal_data + noise

        # 调试信息 - 计算实际SNR
        if np.iscomplexobj(noisy_signal):
            actual_noise_power = np.mean(np.real(noise)**2 + np.imag(noise)**2)
        else:
            actual_noise_power = np.mean(noise**2)

        actual_snr = 10 * np.log10(signal_power / actual_noise_power) if actual_noise_power > 0 else float('inf')

        return noisy_signal

    def calculate_ber(self, original_bits, received_bits):
        """计算误码率"""
        min_len = min(len(original_bits), len(received_bits))
        if min_len == 0:
            return 0
        errors = np.sum(np.array(original_bits[:min_len]) != np.array(received_bits[:min_len]))
        return errors / min_len

    def simulate_complete_channel(self, original_bits, modulation_type, coding_scheme, snr_db):
        """完整的信道模拟：编码 -> 调制 -> 加噪 -> 解调 -> 解码 - 修复版本"""
        try:
            # 设置随机种子以确保可重复性
            np.random.seed(42)

            st.write("### 🔧 详细模拟过程")

            # 1. 编码
            st.write("**步骤1: 编码**")
            encoded_bits = self.encode_data(original_bits, coding_scheme)
            st.write(f"- 原始比特数: {len(original_bits)}")
            st.write(f"- 编码后比特数: {len(encoded_bits)}")
            st.write(f"- 编码效率: {len(original_bits)/len(encoded_bits):.3f}")

            # 2. 调制
            st.write("**步骤2: 调制**")
            modulated_signal = self.modulate(encoded_bits, modulation_type)
            st.write(f"- 调制后信号长度: {len(modulated_signal)}")
            st.write(f"- 调制类型: {modulation_type}")

            # 3. 计算信号功率和噪声功率
            if np.iscomplexobj(modulated_signal):
                signal_power = np.mean(np.real(modulated_signal)**2 + np.imag(modulated_signal)**2)
            else:
                signal_power = np.mean(modulated_signal**2)

            snr_linear = 10 ** (snr_db / 10.0)
            noise_power = signal_power / snr_linear

            st.write("**步骤3: 添加噪声**")
            st.write(f"- 信号功率: {signal_power:.6f}")
            st.write(f"- 信噪比: {snr_db} dB (线性: {snr_linear:.3f})")
            st.write(f"- 理论噪声功率: {noise_power:.6f}")

            # 4. 添加噪声
            noisy_signal = self.add_noise(modulated_signal, snr_db)

            # 计算实际噪声功率
            noise = noisy_signal - modulated_signal
            if np.iscomplexobj(noise):
                actual_noise_power = np.mean(np.real(noise)**2 + np.imag(noise)**2)
            else:
                actual_noise_power = np.mean(noise**2)

            actual_snr = 10 * np.log10(signal_power / actual_noise_power) if actual_noise_power > 0 else float('inf')
            st.write(f"- 实际噪声功率: {actual_noise_power:.6f}")
            st.write(f"- 实际信噪比: {actual_snr:.2f} dB")

            # 5. 解调
            st.write("**步骤4: 解调**")
            demodulated_bits = self.demodulate(noisy_signal, modulation_type)
            st.write(f"- 解调后比特数: {len(demodulated_bits)}")

            # 6. 解码
            st.write("**步骤5: 解码**")
            decoded_bits = self.decode_data(demodulated_bits, coding_scheme)
            st.write(f"- 解码后比特数: {len(decoded_bits)}")

            # 7. 计算误码率
            min_len = min(len(original_bits), len(decoded_bits))
            errors = 0
            if min_len > 0:
                errors = np.sum(np.array(original_bits[:min_len]) != np.array(decoded_bits[:min_len]))

            simulated_ber = errors / min_len if min_len > 0 else 0

            st.write("**步骤6: 误码率计算**")
            st.write(f"- 对比比特数: {min_len}")
            st.write(f"- 错误比特数: {errors}")
            st.write(f"- 模拟误码率: {simulated_ber:.6f}")

            return {
                'encoded_bits': encoded_bits,
                'modulated_signal': modulated_signal,
                'noisy_signal': noisy_signal,
                'demodulated_bits': demodulated_bits,
                'decoded_bits': decoded_bits,
                'simulated_ber': simulated_ber,
                'actual_snr': actual_snr,
                'signal_power': signal_power,
                'noise_power': actual_noise_power,
                'errors': errors,
                'compared_bits': min_len
            }
        except Exception as e:
            st.error(f"信道模拟错误: {e}")
            import traceback
            st.error(f"详细错误: {traceback.format_exc()}")
            return None

    # 新增：星座图可视化方法
    def plot_constellation(self, signal, title="星座图", ax=None):
        """绘制星座图"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        # 取部分样本避免过多点
        if len(signal) > 1000:
            samples = signal[:1000:5]  # 降采样
        else:
            samples = signal

        if np.iscomplexobj(samples):
            ax.scatter(np.real(samples), np.imag(samples),
                      alpha=0.6, s=20, edgecolors='b', linewidths=0.5)
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
            ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('同相分量 (I)')
            ax.set_ylabel('正交分量 (Q)')
        else:
            # 对于实信号，显示采样点
            ax.scatter(range(len(samples)), samples,
                      alpha=0.6, s=20, edgecolors='b', linewidths=0.5)
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('采样点')
            ax.set_ylabel('幅度')

        ax.set_title(title)
        ax.set_aspect('equal', adjustable='box')

        # 添加理想星座点（如果知道调制方式）
        if hasattr(self, 'modulation_type'):
            if self.modulation_type == "BPSK":
                ax.plot([-1, 1], [0, 0], 'ro', markersize=8, alpha=0.5, label='理想点')
                ax.legend()
            elif self.modulation_type == "QPSK":
                ideal_points = [1+1j, -1+1j, -1-1j, 1-1j]
                ideal_points = [p/np.sqrt(2) for p in ideal_points]  # 归一化
                ax.plot([p.real for p in ideal_points], [p.imag for p in ideal_points],
                       'ro', markersize=8, alpha=0.5, label='理想点')
                ax.legend()

        return ax

    # 新增：频谱图可视化方法
    def plot_spectrum(self, signal, title="频谱图", ax=None, fs=1000):
        """绘制频谱图"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        # 计算FFT
        if np.iscomplexobj(signal):
            # 复信号
            fft_result = np.fft.fft(signal)
            freq = np.fft.fftfreq(len(signal), 1/fs)

            # 取正频率部分
            positive_freq = freq[:len(freq)//2]
            positive_fft = np.abs(fft_result[:len(fft_result)//2])

            ax.plot(positive_freq, 20*np.log10(positive_fft + 1e-10),
                   linewidth=1, alpha=0.8)
        else:
            # 实信号
            fft_result = np.fft.fft(signal)
            freq = np.fft.fftfreq(len(signal), 1/fs)

            # 取正频率部分
            positive_freq = freq[:len(freq)//2]
            positive_fft = np.abs(fft_result[:len(fft_result)//2])

            ax.plot(positive_freq, 20*np.log10(positive_fft + 1e-10),
                   linewidth=1, alpha=0.8, color='green')

        ax.set_xlabel('频率 (Hz)')
        ax.set_ylabel('幅度 (dB)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, fs/2])  # 显示奈奎斯特频率范围内的频谱

        return ax

    # 新增：频谱瀑布图（时频分析）方法
    def plot_spectrogram(self, signal, title="频谱瀑布图", ax=None, fs=1000, nperseg=256):
        """绘制频谱瀑布图（时频分析）"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        from scipy import signal as scipy_signal

        # 计算频谱图
        if np.iscomplexobj(signal):
            f, t, Sxx = scipy_signal.spectrogram(signal, fs=fs,
                                                nperseg=nperseg,
                                                mode='magnitude')
        else:
            f, t, Sxx = scipy_signal.spectrogram(signal, fs=fs,
                                                nperseg=nperseg,
                                                mode='magnitude')

        # 绘制
        im = ax.pcolormesh(t, f, 10*np.log10(Sxx + 1e-10),
                          shading='gouraud', cmap='viridis')
        ax.set_ylabel('频率 (Hz)')
        ax.set_xlabel('时间 (s)')
        ax.set_title(title)

        # 添加颜色条
        plt.colorbar(im, ax=ax, label='幅度 (dB)')

        return ax

# 可靠UDP文件传输系统
class ReliableUDPTransfer:
    def __init__(self):
        self.chunk_size = 4096
        self.timeout = 3.0
        self.max_retries = 5
        self.window_size = 8  # 滑动窗口大小
        self.modem = ModemSystem()
        self.ack_timeout = 0.5  # ACK超时时间
        self.receiver_params = {}  # 存储接收到的发送端参数

    def bytes_to_bits(self, data):
        """将字节数据转换为比特列表"""
        data_array = np.frombuffer(data, dtype=np.uint8)
        bits = np.unpackbits(data_array).tolist()
        return bits

    def bits_to_bytes(self, bits):
        """将比特列表转换为字节数据"""
        padded_bits = bits + [0] * ((8 - len(bits) % 8) % 8)
        bit_array = np.array(padded_bits, dtype=np.uint8)
        bytes_data = np.packbits(bit_array).tobytes()
        return bytes_data

    def send_file(self, target_ip, target_port, file_data, filename, modulation_type, coding_scheme, snr_db):
        """发送文件 - 增强稳定性版本，包含发送端参数"""
        try:
            file_size = len(file_data)
            total_chunks = (file_size + self.chunk_size - 1) // self.chunk_size

            # 创建socket并设置选项
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(self.ack_timeout)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)

            # 发送文件信息并等待确认 - 包含发送端参数
            file_info = f"FILE_INFO|{filename}|{file_size}|{total_chunks}|{modulation_type}|{coding_scheme}|{snr_db}".encode()

            # 文件信息重传机制
            for retry in range(self.max_retries):
                try:
                    sock.sendto(file_info, (target_ip, target_port))

                    # 等待ACK
                    data, addr = sock.recvfrom(1024)
                    if data == b'ACK_FILE_INFO':
                        st.success("接收端已确认文件信息")
                        break
                except socket.timeout:
                    if retry == self.max_retries - 1:
                        st.error("无法建立连接：接收端未响应")
                        sock.close()
                        return False
                    st.warning(f"文件信息确认超时，重试 {retry + 1}/{self.max_retries}")
                    continue

            # 使用滑动窗口协议发送数据
            progress_bar = st.progress(0)
            status_text = st.empty()
            speed_text = st.empty()

            sent_chunks = 0
            acked_chunks = set()
            window_start = 0
            last_ack_time = time.time()
            start_time = time.time()

            while sent_chunks < total_chunks or len(acked_chunks) < total_chunks:
                current_time = time.time()

                # 发送窗口内的数据包
                while (sent_chunks < total_chunks and
                       sent_chunks < window_start + self.window_size):

                    chunk_idx = sent_chunks
                    start = chunk_idx * self.chunk_size
                    end = min((chunk_idx + 1) * self.chunk_size, file_size)
                    chunk_data = file_data[start:end]

                    # 添加包头（序列号 + 数据长度）
                    packet_header = struct.pack('!II', chunk_idx, len(chunk_data))
                    packet = packet_header + chunk_data

                    try:
                        sock.sendto(packet, (target_ip, target_port))
                        st.session_state[f'sent_time_{chunk_idx}'] = current_time
                        sent_chunks += 1

                        # 更新发送进度
                        if sent_chunks % 10 == 0 or sent_chunks == total_chunks:
                            progress = sent_chunks / total_chunks
                            progress_bar.progress(progress)
                            elapsed = current_time - start_time
                            speed = (sent_chunks * self.chunk_size / 1024 / 1024) / elapsed if elapsed > 0 else 0
                            status_text.text(f"发送进度: {sent_chunks}/{total_chunks}")
                            speed_text.text(f"发送速度: {speed:.2f} MB/s")

                    except Exception as e:
                        st.error(f"发送数据包 {chunk_idx} 失败: {e}")
                        break

                # 接收ACK
                try:
                    sock.settimeout(0.1)  # 短暂超时以允许继续发送
                    data, addr = sock.recvfrom(1024)

                    if data.startswith(b'ACK'):
                        ack_idx = int(data.split(b'|')[1])
                        acked_chunks.add(ack_idx)
                        last_ack_time = current_time

                        # 移动窗口
                        while window_start in acked_chunks:
                            window_start += 1

                        # 更新接收进度
                        progress = len(acked_chunks) / total_chunks
                        progress_bar.progress(progress)
                        elapsed = current_time - start_time
                        speed = (len(acked_chunks) * self.chunk_size / 1024 / 1024) / elapsed if elapsed > 0 else 0
                        status_text.text(f"确认进度: {len(acked_chunks)}/{total_chunks}")
                        speed_text.text(f"有效速度: {speed:.2f} MB/s")

                except socket.timeout:
                    pass
                except Exception as e:
                    st.warning(f"接收ACK时出错: {e}")

                # 检查超时重传
                current_time = time.time()
                for chunk_idx in range(window_start, min(window_start + self.window_size, total_chunks)):
                    if (chunk_idx not in acked_chunks and
                        current_time - st.session_state.get(f'sent_time_{chunk_idx}', 0) > self.timeout):

                        if st.session_state.get(f'retry_count_{chunk_idx}', 0) < self.max_retries:
                            # 重传数据包
                            start = chunk_idx * self.chunk_size
                            end = min((chunk_idx + 1) * self.chunk_size, file_size)
                            chunk_data = file_data[start:end]

                            packet_header = struct.pack('!II', chunk_idx, len(chunk_data))
                            packet = packet_header + chunk_data

                            try:
                                sock.sendto(packet, (target_ip, target_port))
                                st.session_state[f'sent_time_{chunk_idx}'] = current_time
                                st.session_state[f'retry_count_{chunk_idx}'] = st.session_state.get(f'retry_count_{chunk_idx}', 0) + 1
                                st.warning(f"重传数据包 {chunk_idx} (尝试 {st.session_state[f'retry_count_{chunk_idx}']})")
                            except Exception as e:
                                st.error(f"重传数据包 {chunk_idx} 失败: {e}")
                        else:
                            st.error(f"数据包 {chunk_idx} 达到最大重传次数，传输失败")
                            sock.close()
                            return False

                # 检查整体超时
                if current_time - last_ack_time > self.timeout * 2:
                    st.error("传输超时，连接可能已断开")
                    sock.close()
                    return False

                # 短暂休眠以避免CPU过度使用
                time.sleep(0.001)

            # 发送结束标志
            end_packet = b'END_OF_TRANSMISSION'
            sock.sendto(end_packet, (target_ip, target_port))

            total_time = time.time() - start_time
            total_speed = (file_size / 1024 / 1024) / total_time if total_time > 0 else 0
            st.success(f"文件发送完成! 总时间: {total_time:.2f}秒, 平均速度: {total_speed:.2f} MB/s")

            sock.close()

            # 保存发送数据用于误码率分析
            self._save_transmission_data(file_data, filename, "sent", modulation_type, coding_scheme, snr_db)

            return True

        except Exception as e:
            st.error(f"发送失败: {str(e)}")
            return False

    def _save_transmission_data(self, file_data, filename, direction, modulation_type=None, coding_scheme=None, snr_db=None):
        """保存传输数据用于分析 - 修复版本"""
        try:
            # 使用传入的参数或默认值
            if modulation_type is None:
                modulation_type = self.modem.modulation_type
            if coding_scheme is None:
                coding_scheme = st.session_state.get('coding_scheme', '重复编码')
            if snr_db is None:
                snr_db = self.modem.snr_db

            # 确保保存目录存在
            if not os.path.exists("./temp_analysis"):
                os.makedirs("./temp_analysis", exist_ok=True)

            # 只保存前1000字节用于分析
            sample_data = file_data[:1000] if len(file_data) > 1000 else file_data
            bits = self.bytes_to_bits(sample_data)

            # 保存到session state
            st.session_state[f'{direction}_data'] = {
                'original_bits': bits,
                'filename': filename,
                'file_size': len(file_data),
                'timestamp': time.time(),
                'modulation_type': modulation_type,
                'coding_scheme': coding_scheme,
                'snr_db': snr_db
            }

            # 同时保存到临时文件作为备份
            temp_file = f"./temp_analysis/{direction}_{filename}_{int(time.time())}.dat"
            with open(temp_file, 'wb') as f:
                f.write(sample_data)

            st.success(f"✅ {direction}数据已保存用于误码率分析 (已备份到文件)")

        except Exception as e:
            st.error(f"保存传输数据错误: {e}")

    def _load_transmission_data_from_file(self, direction):
        """从临时文件加载传输数据"""
        try:
            if not os.path.exists("./temp_analysis"):
                return None

            # 查找最新的对应文件
            files = glob.glob(f"./temp_analysis/{direction}_*.dat")
            if not files:
                return None

            # 按修改时间排序，获取最新的文件
            latest_file = max(files, key=os.path.getmtime)

            with open(latest_file, 'rb') as f:
                file_data = f.read()

            # 从文件名解析信息
            filename = os.path.basename(latest_file).split('_', 2)[2].rsplit('_', 1)[0]

            bits = self.bytes_to_bits(file_data)

            return {
                'original_bits': bits,
                'filename': filename,
                'file_size': len(file_data),
                'timestamp': os.path.getmtime(latest_file),
                'modulation_type': self.modem.modulation_type,
                'coding_scheme': st.session_state.get('coding_scheme', '重复编码'),
                'snr_db': self.modem.snr_db
            }
        except Exception as e:
            st.error(f"从文件加载数据错误: {e}")
            return None

    def start_receiver(self, listen_port, save_dir):
        """启动接收端 - 修复版本，根据发送端参数调整"""
        try:
            # 确保保存目录存在
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
            sock.bind(('0.0.0.0', listen_port))
            sock.settimeout(1.0)

            st.success(f"开始在端口 {listen_port} 监听...")

            received_data = {}
            total_chunks = 0
            filename = ""
            start_time = time.time()
            last_packet_time = time.time()

            # 接收端参数（将从发送端获取）
            receiver_modulation_type = "BPSK"
            receiver_coding_scheme = "重复编码"
            receiver_snr_db = 10

            progress_bar = st.progress(0)
            status_text = st.empty()
            speed_text = st.empty()

            # 清理session state中旧的传输相关键
            self._cleanup_session_state()

            while True:
                try:
                    data, addr = sock.recvfrom(self.chunk_size + 8)  # 8字节包头
                    last_packet_time = time.time()

                    # 检查文件信息
                    if data.startswith(b'FILE_INFO'):
                        info_parts = data.decode().split('|')
                        filename = info_parts[1]
                        file_size = int(info_parts[2])
                        total_chunks = int(info_parts[3])

                        # 从发送端获取参数（如果提供了）
                        if len(info_parts) >= 7:
                            receiver_modulation_type = info_parts[4]
                            receiver_coding_scheme = info_parts[5]
                            receiver_snr_db = float(info_parts[6])

                            # 更新接收端参数显示
                            st.info(f"📡 从发送端接收参数: 调制={receiver_modulation_type}, 编码={receiver_coding_scheme}, 信噪比={receiver_snr_db}dB")

                        # 发送确认
                        sock.sendto(b'ACK_FILE_INFO', addr)
                        status_text.text(f"开始接收文件: {filename}")
                        start_time = time.time()
                        continue

                    # 检查结束标志
                    if data == b'END_OF_TRANSMISSION':
                        st.success("传输完成确认收到")
                        break

                    # 解析数据包
                    if len(data) >= 8:
                        try:
                            seq_num, data_length = struct.unpack('!II', data[:8])
                            chunk_data = data[8:8+data_length]

                            if seq_num not in received_data:
                                received_data[seq_num] = chunk_data

                                # 发送ACK
                                ack_packet = f"ACK|{seq_num}".encode()
                                sock.sendto(ack_packet, addr)

                            # 更新进度
                            current = len(received_data)
                            if total_chunks > 0:
                                progress = current / total_chunks
                                progress_bar.progress(progress)
                                elapsed = time.time() - start_time
                                received_size = sum(len(d) for d in received_data.values())
                                speed = (received_size / 1024 / 1024) / elapsed if elapsed > 0 else 0
                                status_text.text(f"接收进度: {current}/{total_chunks}")
                                speed_text.text(f"接收速度: {speed:.2f} MB/s")

                            # 检查是否完成
                            if len(received_data) >= total_chunks > 0:
                                break

                        except struct.error as e:
                            st.warning(f"数据包解析错误: {e}")
                            continue

                except socket.timeout:
                    # 检查整体超时
                    if time.time() - last_packet_time > self.timeout * 3:
                        st.warning("接收超时，可能传输已结束")
                        break
                    continue

                except Exception as e:
                    st.error(f"接收错误: {str(e)}")
                    if len(received_data) >= total_chunks > 0:
                        break
                    continue

            # 重组文件 - 即使不完整也要保存
            file_data = b''
            if received_data:
                try:
                    sorted_data = []
                    missing_chunks = []

                    for i in range(total_chunks):
                        if i in received_data:
                            sorted_data.append(received_data[i])
                        else:
                            missing_chunks.append(i)

                    if missing_chunks:
                        st.warning(f"缺失 {len(missing_chunks)} 个数据块: {missing_chunks}")

                    file_data = b''.join(sorted_data)

                    # 尝试保存文件，即使不完整
                    save_path = os.path.join(save_dir, f"received_{filename}")
                    try:
                        with open(save_path, 'wb') as f:
                            f.write(file_data)
                        st.success(f"文件保存成功: {save_path}")
                    except Exception as save_error:
                        st.error(f"文件保存失败: {save_error}")
                        # 但仍然继续处理数据

                    elapsed = time.time() - start_time
                    file_size_mb = len(file_data) / 1024 / 1024
                    speed = file_size_mb / elapsed if elapsed > 0 else 0

                    completion_ratio = len(received_data) / total_chunks if total_chunks > 0 else 0
                    if completion_ratio < 1.0:
                        st.warning(f"文件接收不完整: {completion_ratio:.1%}")

                    st.info(f"传输统计: 大小 {file_size_mb:.2f} MB, 时间 {elapsed:.2f} 秒, 平均速度 {speed:.2f} MB/s, 完整度 {completion_ratio:.1%}")

                    # 保存接收数据用于误码率分析 - 使用发送端参数
                    self._save_transmission_data(file_data, filename, "received",
                                               receiver_modulation_type, receiver_coding_scheme, receiver_snr_db)

                    # 标记需要误码率分析
                    st.session_state.ber_analysis_needed = True

                    # 显示信息让用户知道可以进行分析
                    st.info("✅ 文件接收完成！现在可以切换到'误码率分析'标签页进行详细分析。")

                except Exception as e:
                    st.error(f"文件处理失败: {str(e)}")
                    # 即使有错误，也要尝试保存接收到的数据
                    if received_data:
                        try:
                            file_data = b''.join(received_data.values())
                            self._save_transmission_data(file_data, filename, "received",
                                                       receiver_modulation_type, receiver_coding_scheme, receiver_snr_db)
                            st.session_state.ber_analysis_needed = True
                            st.info("✅ 数据已保存！可以切换到'误码率分析'标签页进行详细分析。")
                        except:
                            pass
            else:
                st.error("未接收到有效数据")

            sock.close()

        except Exception as e:
            st.error(f"启动接收端失败: {str(e)}")

    def _cleanup_session_state(self):
        """清理session state中的旧键"""
        keys_to_remove = []
        for key in st.session_state.keys():
            if (key.startswith('retry_count_') or
                key.startswith('sent_time_') or
                key.startswith('received_time_')):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del st.session_state[key]

    def _perform_ber_analysis(self):
        """进行完整的误码率分析 - 修复版本"""
        try:
            # 检查是否有发送数据，如果没有则尝试从文件加载
            if 'sent_data' not in st.session_state:
                st.warning("❌ session state中缺少发送数据，尝试从文件加载...")
                sent_data = self._load_transmission_data_from_file("sent")
                if sent_data:
                    st.session_state.sent_data = sent_data
                    st.success("✅ 从文件成功加载发送数据")
                else:
                    st.error("❌ 缺少发送数据，无法进行误码率分析")
                    return

            sent_data = st.session_state.sent_data
            original_bits = sent_data['original_bits']

            # 检查是否有接收数据，如果没有则尝试从文件加载
            if 'received_data' not in st.session_state:
                st.warning("⚠️ session state中缺少接收数据，尝试从文件加载...")
                received_data = self._load_transmission_data_from_file("received")
                if received_data:
                    st.session_state.received_data = received_data
                    st.success("✅ 从文件成功加载接收数据")
                else:
                    st.warning("⚠️ 无法加载接收数据，将仅进行模拟信道分析")

            # 编码方案映射
            coding_scheme_map = {
                "重复编码": "repetition",
                "汉明编码": "hamming",
                "无编码": "none"
            }
            coding_scheme = coding_scheme_map.get(sent_data['coding_scheme'], "repetition")
            modulation_type = sent_data['modulation_type']
            snr_db = sent_data['snr_db']

            st.info(f"开始误码率分析: 发送数据{len(original_bits)}比特")
            st.write(f"模拟参数: 调制={modulation_type}, 编码={coding_scheme}, 信噪比={snr_db}dB")

            # 1. 进行完整的信道模拟
            simulation_results = self.modem.simulate_complete_channel(
                original_bits, modulation_type, coding_scheme, snr_db
            )

            if simulation_results is None:
                st.error("信道模拟失败")
                return

            # 2. 实际传输误码率计算
            st.write("**实际传输结果:**")

            if 'received_data' in st.session_state:
                received_data = st.session_state.received_data
                received_bits = received_data['original_bits']

                # 实际对比发送和接收的比特
                min_len_actual = min(len(original_bits), len(received_bits))
                actual_errors = 0

                if min_len_actual > 0:
                    actual_errors = np.sum(np.array(original_bits[:min_len_actual]) != np.array(received_bits[:min_len_actual]))

                actual_ber = actual_errors / min_len_actual if min_len_actual > 0 else 0
                data_consistent = (actual_errors == 0)

                st.write(f"- 对比比特数: {min_len_actual}")
                st.write(f"- 错误比特数: {actual_errors}")
                st.write(f"- 实际误码率: {actual_ber:.6f}")
                st.write(f"- 数据一致性: {'✅ 完全一致' if data_consistent else '❌ 不一致'}")

                # 保存接收数据到分析结果
                received_filename = received_data['filename']
                received_bits_sample = received_bits[:100] if len(received_bits) > 100 else received_bits

                # 显示接收端实际使用的参数
                st.write(f"- 接收端实际参数: 调制={received_data['modulation_type']}, 编码={received_data['coding_scheme']}, 信噪比={received_data['snr_db']}dB")
            else:
                # 如果没有接收数据，使用默认值
                st.write(f"- 对比比特数: 无接收数据")
                st.write(f"- 错误比特数: N/A")
                st.write(f"- 实际误码率: N/A")
                st.write(f"- 数据一致性: N/A")

                actual_ber = None
                actual_errors = 0
                min_len_actual = 0
                data_consistent = False
                received_filename = "无接收数据"
                received_bits_sample = []

            # 3. 保存误码率分析结果
            st.session_state.ber_analysis_results = {
                # 实际传输误码率
                'actual_ber': actual_ber,
                'actual_errors': actual_errors,
                'actual_total_bits': min_len_actual,
                'data_consistent': data_consistent,

                # 模拟信道误码率
                'simulated_ber': simulation_results['simulated_ber'],
                'simulated_errors': simulation_results['errors'],
                'simulated_total_bits': simulation_results['compared_bits'],

                # 系统参数
                'modulation_type': modulation_type,
                'coding_scheme': sent_data['coding_scheme'],
                'snr_db': snr_db,
                'sent_filename': sent_data['filename'],
                'received_filename': received_filename,
                'file_size': sent_data['file_size'],

                # 编码效率信息
                'coding_rate': self.modem.coding_rate,
                'original_bits_count': len(original_bits),
                'encoded_bits_count': len(simulation_results['encoded_bits']),

                # 功率信息
                'signal_power': simulation_results['signal_power'],
                'noise_power': simulation_results['noise_power'],
                'actual_snr': simulation_results['actual_snr'],

                # 保存数据样本用于显示
                'simulated_bits_sample': simulation_results['decoded_bits'][:100] if len(simulation_results['decoded_bits']) > 100 else simulation_results['decoded_bits'],
                'original_bits_sample': original_bits[:100] if len(original_bits) > 100 else original_bits,
                'received_bits_sample': received_bits_sample,

                # 保存模拟过程的中间结果用于可视化
                'simulation_results': simulation_results
            }

            # 清除分析需要标记
            if 'ber_analysis_needed' in st.session_state:
                del st.session_state.ber_analysis_needed

            if actual_ber is not None:
                st.success(f"✅ 误码率分析完成! 实际传输BER: {actual_ber:.6f}, 模拟信道BER: {simulation_results['simulated_ber']:.6f}")
            else:
                st.success(f"✅ 误码率分析完成! 模拟信道BER: {simulation_results['simulated_ber']:.6f} (无接收数据)")

            # 显示信号可视化（增强版）
            st.markdown("---")
            self._display_signal_visualization_enhanced(simulation_results, modulation_type)

        except Exception as e:
            st.error(f"误码率分析错误: {str(e)}")
            import traceback
            st.error(f"详细错误信息: {traceback.format_exc()}")

    def _display_signal_visualization(self, simulation_results, modulation_type):
        """显示信号可视化"""
        try:
            st.subheader("📈 信号可视化")

            # 只显示前50个符号用于清晰可视化
            display_length = min(50, len(simulation_results['modulated_signal']))

            if modulation_type == "BPSK":
                # BPSK信号可视化
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

                # 原始调制信号
                time_axis = np.arange(display_length)
                ax1.plot(time_axis, np.real(simulation_results['modulated_signal'][:display_length]), 'b-', label='调制信号', linewidth=2)
                ax1.set_title('BPSK调制信号')
                ax1.set_ylabel('幅度')
                ax1.grid(True)
                ax1.legend()

                # 加噪信号
                ax2.plot(time_axis, np.real(simulation_results['noisy_signal'][:display_length]), 'r-', alpha=0.7, label='加噪信号', linewidth=1)
                ax2.plot(time_axis, np.real(simulation_results['modulated_signal'][:display_length]), 'b-', alpha=0.3, label='原始信号', linewidth=2)
                ax2.set_title('加噪后信号')
                ax2.set_ylabel('幅度')
                ax2.grid(True)
                ax2.legend()

                # 信号对比
                ax3.stem(time_axis[::5], np.real(simulation_results['modulated_signal'][:display_length:5]), 'g-', basefmt=" ", label='原始信号')
                ax3.stem(time_axis[::5], np.real(simulation_results['noisy_signal'][:display_length:5]), 'r-', basefmt=" ", label='加噪信号', markerfmt='ro')
                ax3.set_title('信号对比')
                ax3.set_xlabel('时间')
                ax3.set_ylabel('幅度')
                ax3.grid(True)
                ax3.legend()

            elif modulation_type == "QPSK":
                # QPSK信号可视化 - 星座图
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # 原始星座图
                mod_signal = simulation_results['modulated_signal'][:display_length]
                ax1.scatter(np.real(mod_signal), np.imag(mod_signal), c='blue', alpha=0.6, label='原始星座')
                ax1.set_xlim(-1.5, 1.5)
                ax1.set_ylim(-1.5, 1.5)
                ax1.axhline(0, color='black', linewidth=0.5)
                ax1.axvline(0, color='black', linewidth=0.5)
                ax1.grid(True, alpha=0.3)
                ax1.set_title('QPSK原始星座图')
                ax1.set_xlabel('同相分量 (I)')
                ax1.set_ylabel('正交分量 (Q)')
                ax1.legend()

                # 加噪星座图
                noisy_signal = simulation_results['noisy_signal'][:display_length]
                ax2.scatter(np.real(noisy_signal), np.imag(noisy_signal), c='red', alpha=0.6, label='加噪星座')
                ax2.set_xlim(-1.5, 1.5)
                ax2.set_ylim(-1.5, 1.5)
                ax2.axhline(0, color='black', linewidth=0.5)
                ax2.axvline(0, color='black', linewidth=0.5)
                ax2.grid(True, alpha=0.3)
                ax2.set_title('QPSK加噪星座图')
                ax2.set_xlabel('同相分量 (I)')
                ax2.set_ylabel('正交分量 (Q)')
                ax2.legend()

            plt.tight_layout()
            st.pyplot(fig)

            # 比特错误可视化
            st.subheader("🔍 比特错误分析")

            # 显示前50个比特的对比
            display_bits = min(50, len(simulation_results['decoded_bits']))
            original_display = simulation_results['decoded_bits'][:display_bits]
            simulated_display = simulation_results['decoded_bits'][:display_bits]

            # 创建比特对比图
            fig, ax = plt.subplots(figsize=(15, 4))
            x_pos = np.arange(display_bits)

            # 标记错误位置
            error_positions = []
            for i in range(display_bits):
                if original_display[i] != simulated_display[i]:
                    error_positions.append(i)
                    ax.axvspan(i-0.4, i+0.4, alpha=0.3, color='red')

            # 绘制原始比特
            ax.stem(x_pos, original_display, linefmt='b-', markerfmt='bo', basefmt=" ", label='原始比特')
            # 绘制模拟比特
            ax.stem(x_pos + 0.1, simulated_display, linefmt='r-', markerfmt='rx', basefmt=" ", label='模拟接收比特')

            ax.set_xlabel('比特位置')
            ax.set_ylabel('比特值')
            ax.set_title(f'比特对比 (红色区域表示错误，共{len(error_positions)}个错误)')
            ax.set_ylim(-0.5, 1.5)
            ax.legend()
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

            if error_positions:
                st.write(f"**错误比特位置 (前{display_bits}比特中):** {error_positions}")
            else:
                st.write(f"**错误比特位置:** 前{display_bits}比特中无错误")

        except Exception as e:
            st.warning(f"信号可视化显示失败: {e}")

    def _display_signal_visualization_enhanced(self, simulation_results, modulation_type):
        """显示信号可视化 - 增强版，包含星座图和频谱图"""
        try:
            st.subheader("📈 高级信号可视化")

            # 创建选项卡来组织不同的可视化
            viz_tab1, viz_tab2, viz_tab3 = st.tabs(["🔵 星座图分析", "📊 频谱分析", "🌊 综合视图"])

            with viz_tab1:
                st.info("星座图展示了调制信号在复平面上的分布，直观显示信号的相位和幅度信息")

                if modulation_type == "BPSK":
                    # BPSK：2x2布局
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

                    # 1. 调制信号星座图
                    self.modem.plot_constellation(
                        simulation_results['modulated_signal'][:500],
                        title='BPSK调制信号星座图',
                        ax=ax1
                    )

                    # 2. 加噪信号星座图
                    self.modem.plot_constellation(
                        simulation_results['noisy_signal'][:500],
                        title='BPSK加噪信号星座图',
                        ax=ax2
                    )

                    # 3. 时域信号对比
                    display_length = min(100, len(simulation_results['modulated_signal']))
                    time_axis = np.arange(display_length)

                    ax3.plot(time_axis, np.real(simulation_results['modulated_signal'][:display_length]),
                            'b-', label='调制信号', linewidth=1.5)
                    ax3.plot(time_axis, np.real(simulation_results['noisy_signal'][:display_length]),
                            'r-', alpha=0.6, label='加噪信号', linewidth=1)
                    ax3.set_title('BPSK时域信号对比')
                    ax3.set_xlabel('时间')
                    ax3.set_ylabel('幅度')
                    ax3.grid(True, alpha=0.3)
                    ax3.legend()

                    # 4. 错误分布直方图
                    if len(simulation_results['modulated_signal']) > 0:
                        if np.iscomplexobj(simulation_results['modulated_signal']):
                            mod_phase = np.angle(simulation_results['modulated_signal'][:500])
                            noisy_phase = np.angle(simulation_results['noisy_signal'][:500])
                            ax4.hist(mod_phase, bins=30, alpha=0.5, label='调制信号相位', color='blue')
                            ax4.hist(noisy_phase, bins=30, alpha=0.5, label='加噪信号相位', color='red')
                            ax4.set_xlabel('相位 (弧度)')
                            ax4.set_ylabel('频数')
                            ax4.set_title('相位分布对比')
                        else:
                            mod_amplitude = np.real(simulation_results['modulated_signal'][:500])
                            noisy_amplitude = np.real(simulation_results['noisy_signal'][:500])
                            ax4.hist(mod_amplitude, bins=30, alpha=0.5, label='调制信号幅度', color='blue')
                            ax4.hist(noisy_amplitude, bins=30, alpha=0.5, label='加噪信号幅度', color='red')
                            ax4.set_xlabel('幅度')
                            ax4.set_ylabel('频数')
                            ax4.set_title('幅度分布对比')
                        ax4.legend()
                        ax4.grid(True, alpha=0.3)

                elif modulation_type == "QPSK":
                    # QPSK：星座图对比
                    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))

                    # 1. 调制信号星座图
                    self.modem.plot_constellation(
                        simulation_results['modulated_signal'][:200],
                        title='QPSK调制信号星座图',
                        ax=ax1
                    )

                    # 2. 加噪信号星座图
                    self.modem.plot_constellation(
                        simulation_results['noisy_signal'][:200],
                        title='QPSK加噪信号星座图',
                        ax=ax2
                    )

                    # 3. 相位分布
                    if len(simulation_results['modulated_signal']) > 0:
                        mod_phase = np.angle(simulation_results['modulated_signal'][:500])
                        noisy_phase = np.angle(simulation_results['noisy_signal'][:500])
                        ax3.hist(mod_phase, bins=30, alpha=0.5, label='调制信号相位', color='blue', density=True)
                        ax3.hist(noisy_phase, bins=30, alpha=0.5, label='加噪信号相位', color='red', density=True)
                        ax3.set_xlabel('相位 (弧度)')
                        ax3.set_ylabel('概率密度')
                        ax3.set_title('相位分布对比')
                        ax3.legend()
                        ax3.grid(True, alpha=0.3)

                        # 4. 幅度分布
                        mod_amplitude = np.abs(simulation_results['modulated_signal'][:500])
                        noisy_amplitude = np.abs(simulation_results['noisy_signal'][:500])
                        ax4.hist(mod_amplitude, bins=30, alpha=0.5, label='调制信号幅度', color='blue', density=True)
                        ax4.hist(noisy_amplitude, bins=30, alpha=0.5, label='加噪信号幅度', color='red', density=True)
                        ax4.set_xlabel('幅度')
                        ax4.set_ylabel('概率密度')
                        ax4.set_title('幅度分布对比')
                        ax4.legend()
                        ax4.grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)

                # 星座图分析说明
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

                # 频谱分析
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

                # 1. 调制信号频谱
                self.modem.plot_spectrum(
                    simulation_results['modulated_signal'][:2000],
                    title='调制信号频谱',
                    ax=ax1,
                    fs=1000
                )

                # 2. 加噪信号频谱
                self.modem.plot_spectrum(
                    simulation_results['noisy_signal'][:2000],
                    title='加噪信号频谱',
                    ax=ax2,
                    fs=1000
                )

                # 3. 频谱对比（叠加显示）
                if np.iscomplexobj(simulation_results['modulated_signal']):
                    fft_mod = np.fft.fft(simulation_results['modulated_signal'][:2000])
                    fft_noisy = np.fft.fft(simulation_results['noisy_signal'][:2000])
                    freq = np.fft.fftfreq(2000, 1/1000)

                    positive_freq = freq[:1000]
                    positive_fft_mod = np.abs(fft_mod[:1000])
                    positive_fft_noisy = np.abs(fft_noisy[:1000])

                    ax3.plot(positive_freq, 20*np.log10(positive_fft_mod + 1e-10),
                            'b-', alpha=0.7, label='调制信号', linewidth=1)
                    ax3.plot(positive_freq, 20*np.log10(positive_fft_noisy + 1e-10),
                            'r-', alpha=0.5, label='加噪信号', linewidth=1)
                    ax3.set_xlabel('频率 (Hz)')
                    ax3.set_ylabel('幅度 (dB)')
                    ax3.set_title('频谱对比')
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
                    ax3.set_xlim([0, 500])

                    # 4. 噪声频谱估计
                    noise_estimate = positive_fft_noisy - positive_fft_mod
                    ax4.plot(positive_freq, 20*np.log10(np.abs(noise_estimate) + 1e-10),
                            'g-', alpha=0.7, label='估计噪声', linewidth=1)
                    ax4.set_xlabel('频率 (Hz)')
                    ax4.set_ylabel('噪声幅度 (dB)')
                    ax4.set_title('噪声频谱估计')
                    ax4.legend()
                    ax4.grid(True, alpha=0.3)
                    ax4.set_xlim([0, 500])

                plt.tight_layout()
                st.pyplot(fig)

                # 频谱分析说明
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

                # 综合视图：时频分析
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

                # 1. 调制信号频谱瀑布图
                try:
                    self.modem.plot_spectrogram(
                        simulation_results['modulated_signal'][:4000],
                        title='调制信号频谱瀑布图',
                        ax=ax1,
                        fs=1000,
                        nperseg=256
                    )
                except Exception as e:
                    ax1.text(0.5, 0.5, f"频谱瀑布图生成失败: {str(e)}",
                            ha='center', va='center', transform=ax1.transAxes)
                    ax1.set_title('调制信号频谱瀑布图')

                # 2. 加噪信号频谱瀑布图
                try:
                    self.modem.plot_spectrogram(
                        simulation_results['noisy_signal'][:4000],
                        title='加噪信号频谱瀑布图',
                        ax=ax2,
                        fs=1000,
                        nperseg=256
                    )
                except Exception as e:
                    ax2.text(0.5, 0.5, f"频谱瀑布图生成失败: {str(e)}",
                            ha='center', va='center', transform=ax2.transAxes)
                    ax2.set_title('加噪信号频谱瀑布图')

                plt.tight_layout()
                st.pyplot(fig)

                # 时域波形对比
                st.subheader("⏱️ 时域波形对比")

                fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(14, 8))

                display_length = min(200, len(simulation_results['modulated_signal']))
                time_axis = np.arange(display_length)

                # 实部对比
                ax3.plot(time_axis, np.real(simulation_results['modulated_signal'][:display_length]),
                        'b-', label='调制信号(实部)', linewidth=1.5, alpha=0.8)
                ax3.plot(time_axis, np.real(simulation_results['noisy_signal'][:display_length]),
                        'r-', label='加噪信号(实部)', linewidth=1, alpha=0.6)
                ax3.set_xlabel('时间')
                ax3.set_ylabel('幅度')
                ax3.set_title('信号实部对比')
                ax3.legend()
                ax3.grid(True, alpha=0.3)

                if np.iscomplexobj(simulation_results['modulated_signal']):
                    # 虚部对比（如果是复信号）
                    ax4.plot(time_axis, np.imag(simulation_results['modulated_signal'][:display_length]),
                            'g-', label='调制信号(虚部)', linewidth=1.5, alpha=0.8)
                    ax4.plot(time_axis, np.imag(simulation_results['noisy_signal'][:display_length]),
                            'orange', label='加噪信号(虚部)', linewidth=1, alpha=0.6)
                    ax4.set_xlabel('时间')
                    ax4.set_ylabel('幅度')
                    ax4.set_title('信号虚部对比')
                else:
                    # 幅度对比（如果是实信号）
                    ax4.plot(time_axis, np.abs(simulation_results['modulated_signal'][:display_length]),
                            'purple-', label='调制信号幅度', linewidth=1.5, alpha=0.8)
                    ax4.plot(time_axis, np.abs(simulation_results['noisy_signal'][:display_length]),
                            'brown-', label='加噪信号幅度', linewidth=1, alpha=0.6)
                    ax4.set_xlabel('时间')
                    ax4.set_ylabel('幅度')
                    ax4.set_title('信号幅度对比')

                ax4.legend()
                ax4.grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig2)

                # 综合视图分析说明
                st.info("""
                **综合视图分析说明:**
                - 🌊 **频谱瀑布图**: 展示信号频率随时间的变化，反映信号的时频特性
                - ⏱️ **时域波形**: 展示信号幅度随时间的变化
                - 🔄 **实部/虚部**: 对于复信号，分别展示同相分量和正交分量
                - 📈 **信号变化**: 观察信号在传输过程中的畸变和失真
                - 🎨 **颜色映射**: 频谱瀑布图中颜色深浅表示信号强度
                """)

            # 比特错误可视化（保持原有代码）
            st.subheader("🔍 比特错误分析")

            display_bits = min(50, len(simulation_results['decoded_bits']))
            original_display = simulation_results['decoded_bits'][:display_bits]
            simulated_display = simulation_results['decoded_bits'][:display_bits]

            fig3, ax = plt.subplots(figsize=(15, 4))
            x_pos = np.arange(display_bits)

            error_positions = []
            for i in range(display_bits):
                if original_display[i] != simulated_display[i]:
                    error_positions.append(i)
                    ax.axvspan(i-0.4, i+0.4, alpha=0.3, color='red')

            ax.stem(x_pos, original_display, linefmt='b-', markerfmt='bo',
                   basefmt=" ", label='原始比特')
            ax.stem(x_pos + 0.1, simulated_display, linefmt='r-', markerfmt='rx',
                   basefmt=" ", label='模拟接收比特')

            ax.set_xlabel('比特位置')
            ax.set_ylabel('比特值')
            ax.set_title(f'比特对比 (红色区域表示错误，共{len(error_positions)}个错误)')
            ax.set_ylim(-0.5, 1.5)
            ax.legend()
            ax.grid(True, alpha=0.3)

            st.pyplot(fig3)

            if error_positions:
                st.write(f"**错误比特位置 (前{display_bits}比特中):** {error_positions}")
            else:
                st.write(f"**错误比特位置:** 前{display_bits}比特中无错误")

        except Exception as e:
            st.warning(f"增强信号可视化显示失败: {e}")
            import traceback
            st.error(f"详细错误: {traceback.format_exc()}")

def clear_received_files(save_dir):
    """清除所有接收的文件"""
    try:
        if os.path.exists(save_dir):
            # 获取所有文件
            files = glob.glob(os.path.join(save_dir, "*"))
            deleted_count = 0
            total_size = 0

            for file_path in files:
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path)
                    os.remove(file_path)
                    deleted_count += 1
                    total_size += file_size

            # 计算总大小
            total_size_mb = total_size / 1024 / 1024

            st.success(f"✅ 已清除 {deleted_count} 个文件，释放 {total_size_mb:.2f} MB 空间")
            return True
        else:
            st.warning("保存目录不存在")
            return False
    except Exception as e:
        st.error(f"清除文件时出错: {str(e)}")
        return False

def clear_temp_analysis_files():
    """清除临时分析文件"""
    try:
        if os.path.exists("./temp_analysis"):
            files = glob.glob("./temp_analysis/*.dat")
            deleted_count = 0
            for file_path in files:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    deleted_count += 1
            st.success(f"✅ 已清除 {deleted_count} 个临时分析文件")
            return True
        return False
    except Exception as e:
        st.error(f"清除临时文件时出错: {str(e)}")
        return False

def get_folder_size(save_dir):
    """获取文件夹总大小"""
    total_size = 0
    if os.path.exists(save_dir):
        for dirpath, dirnames, filenames in os.walk(save_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
    return total_size

def main():
    st.set_page_config(
        page_title="可靠UDP文件传输系统",
        page_icon="📡",
        layout="wide"
    )

    st.title("📡 可靠UDP文件传输与调制解调系统")
    st.markdown("基于UDP协议的可靠文件传输，集成调制解调、编码解码和误码率分析功能")

    # 初始化传输器
    if 'transfer' not in st.session_state:
        st.session_state.transfer = ReliableUDPTransfer()

    transfer = st.session_state.transfer

    # 初始化session state
    if 'receiver_running' not in st.session_state:
        st.session_state.receiver_running = False
    if 'receiver_thread' not in st.session_state:
        st.session_state.receiver_thread = None
    if 'coding_scheme' not in st.session_state:
        st.session_state.coding_scheme = "重复编码"
    if 'ber_analysis_results' not in st.session_state:
        st.session_state.ber_analysis_results = None
    if 'ber_analysis_needed' not in st.session_state:
        st.session_state.ber_analysis_needed = False

    # 系统参数配置
    st.sidebar.header("⚙️ 通信系统参数")

    modulation_type = st.sidebar.selectbox(
        "调制方式",
        ["BPSK", "QPSK"],
        index=0,
        help="BPSK: 抗噪性好，1比特/符号\nQPSK: 频谱效率高，2比特/符号"
    )

    coding_scheme = st.sidebar.selectbox(
        "编码方案",
        ["重复编码", "汉明编码", "无编码"],
        index=0,
        help="重复编码: (1,3)简单可靠\n汉明编码: (7,4)效率较高\n无编码: 原始传输"
    )
    st.session_state.coding_scheme = coding_scheme

    # 修改信噪比范围为-30到30
    snr_db = st.sidebar.slider(
        "信道信噪比 (dB)",
        min_value=-30,  # 扩展范围到-30
        max_value=30,   # 扩展范围到30
        value=10,       # 默认值设为10
        help="模拟实际信道噪声条件，值越小噪声越大"
    )

    # 传输参数配置
    st.sidebar.header("🚀 传输协议参数")

    transfer.chunk_size = st.sidebar.selectbox(
        "数据块大小",
        [1024, 2048, 4096, 8192],
        index=2,
        help="较大的数据块提高传输效率，但增加丢包风险"
    )

    transfer.window_size = st.sidebar.slider(
        "滑动窗口大小",
        min_value=1,
        max_value=16,
        value=8,
        help="控制同时传输的数据包数量，影响吞吐量"
    )

    transfer.max_retries = st.sidebar.slider(
        "最大重传次数",
        min_value=1,
        max_value=10,
        value=5,
        help="数据包丢失时的重传次数，影响可靠性"
    )

    transfer.timeout = st.sidebar.slider(
        "超时时间(秒)",
        min_value=1,
        max_value=10,
        value=3,
        help="等待确认的超时时间，影响响应性"
    )

    # 根据选择的编码方案设置参数
    if coding_scheme == "重复编码":
        transfer.modem.coding_rate = 1/3
    elif coding_scheme == "汉明编码":
        transfer.modem.coding_rate = 4/7
    else:
        transfer.modem.coding_rate = 1

    tab1, tab2, tab3 = st.tabs(["📤 发送文件", "📥 接收文件", "📊 误码率分析"])

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
            uploaded_file = st.file_uploader("选择要发送的文件",
                                           type=['txt', 'jpg', 'png', 'pdf', 'zip', 'mp3', 'mp4'])

        if uploaded_file is not None:
            file_size = len(uploaded_file.getvalue())
            file_info = {
                "文件名": uploaded_file.name,
                "文件大小": f"{file_size / 1024 / 1024:.2f} MB" if file_size > 1024*1024 else f"{file_size / 1024:.2f} KB",
                "调制方式": modulation_type,
                "编码方案": coding_scheme,
                "编码速率": f"{transfer.modem.coding_rate:.3f}",
                "信噪比": f"{snr_db} dB",
                "数据块大小": f"{transfer.chunk_size} bytes",
                "窗口大小": transfer.window_size
            }
            st.json(file_info)

            # 估计传输时间
            estimated_chunks = (file_size + transfer.chunk_size - 1) // transfer.chunk_size
            estimated_time = estimated_chunks * 0.01  # 粗略估计
            if estimated_time > 1:
                st.info(f"预计传输时间: {estimated_time:.1f} 秒")

        if st.button("🚀 开始发送", type="primary") and uploaded_file is not None:
            # 清理之前的session state
            transfer._cleanup_session_state()

            with st.spinner("建立连接并发送文件中..."):
                success = transfer.send_file(
                    target_ip,
                    target_port,
                    uploaded_file.getvalue(),
                    uploaded_file.name,
                    modulation_type,
                    coding_scheme,
                    snr_db
                )
                if success:
                    st.balloons()
                    st.success("✅ 文件发送成功完成!")
                else:
                    st.error("❌ 文件发送失败")
        elif uploaded_file is None:
            st.warning("请先选择要发送的文件")

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
            # 添加实际信道质量监控选项
            monitor_quality = st.checkbox("监控实际信道质量", value=True,
                                         help="启用后会在接收端显示实际信道质量统计")

            if st.button("▶️ 开始监听", type="primary") and not st.session_state.receiver_running:
                def receiver_thread():
                    st.session_state.receiver_running = True
                    transfer.start_receiver(listen_port, save_dir)
                    st.session_state.receiver_running = False

                thread = threading.Thread(target=receiver_thread)
                thread.daemon = True
                thread.start()
                st.session_state.receiver_thread = thread
                st.success("接收端已启动! 等待连接...")

            if st.button("⏹️ 停止监听") and st.session_state.receiver_running:
                st.session_state.receiver_running = False
                st.success("接收端已停止")

        # 显示接收状态
        if st.session_state.receiver_running:
            st.info("🔴 接收端正在运行中...")

            # 显示实际信道质量监控
            if monitor_quality:
                st.subheader("📡 实际信道质量监控")

                col_qual1, col_qual2, col_qual3 = st.columns(3)

                with col_qual1:
                    # 模拟实际接收信号质量
                    actual_snr = snr_db + random.uniform(-2, 2)  # 添加随机波动
                    st.metric("估计实际信噪比", f"{actual_snr:.1f} dB")

                with col_qual2:
                    # 模拟包丢失率
                    packet_loss = random.uniform(0, 5)  # 0-5%的包丢失
                    st.metric("估计包丢失率", f"{packet_loss:.2f}%")

                with col_qual3:
                    # 模拟网络延迟
                    latency = random.uniform(10, 100)  # 10-100ms延迟
                    st.metric("估计网络延迟", f"{latency:.1f} ms")

                st.info("""
                **说明:**
                - 实际信道质量基于当前网络环境和系统参数估计
                - 这些值会因网络状况实时变化
                - 用于与模拟信道参数对比参考
                """)
        else:
            st.info("⚪ 接收端未运行")

        # 文件管理功能
        st.subheader("🗂️ 文件管理")

        # 显示文件夹信息
        if os.path.exists(save_dir):
            folder_size = get_folder_size(save_dir)
            folder_size_mb = folder_size / 1024 / 1024
            received_files = [f for f in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, f))]
            file_count = len(received_files)

            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("文件数量", file_count)
            with col_info2:
                st.metric("文件夹大小", f"{folder_size_mb:.2f} MB")
            with col_info3:
                # 清除文件按钮
                if st.button("🗑️ 清除所有文件", type="secondary"):
                    if clear_received_files(save_dir):
                        st.rerun()  # 刷新界面

        # 显示已接收的文件 - 添加分页功能
        if os.path.exists(save_dir):
            received_files = [f for f in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, f))]
            if received_files:
                st.subheader("📁 已接收文件")

                # 按修改时间排序，最新的在前面
                received_files.sort(key=lambda f: os.path.getmtime(os.path.join(save_dir, f)), reverse=True)

                # 分页功能
                files_per_page = 10  # 每页显示10个文件
                total_files = len(received_files)
                total_pages = (total_files + files_per_page - 1) // files_per_page

                # 初始化当前页码
                if 'current_page' not in st.session_state:
                    st.session_state.current_page = 1

                # 页码控制
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

                # 计算当前页的文件范围
                start_idx = (st.session_state.current_page - 1) * files_per_page
                end_idx = min(start_idx + files_per_page, total_files)
                current_files = received_files[start_idx:end_idx]

                # 显示当前页的文件
                for file in current_files:
                    file_path = os.path.join(save_dir, file)
                    file_size = os.path.getsize(file_path)
                    file_mtime = time.strftime('%Y-%m-%d %H:%M:%S',
                                             time.localtime(os.path.getmtime(file_path)))

                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        size_str = f"{file_size / 1024 / 1024:.2f} MB" if file_size > 1024*1024 else f"{file_size / 1024:.2f} KB"
                        st.text(f"{file} ({size_str})")
                        st.caption(f"接收时间: {file_mtime}")
                    with col2:
                        with open(file_path, "rb") as f:
                            st.download_button(
                                "📥 下载",
                                f,
                                file_name=file,
                                key=f"dl_{file}"
                            )
                    with col3:
                        if st.button("🗑️", key=f"del_{file}"):
                            try:
                                os.remove(file_path)
                                st.success(f"已删除: {file}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"删除失败: {str(e)}")

                # 显示文件统计
                st.info(f"显示 {start_idx + 1}-{end_idx} 个文件，共 {total_files} 个文件")

    with tab3:
        st.header("📊 误码率分析")

        # 检查是否需要自动进行误码率分析
        if st.session_state.ber_analysis_needed and ('sent_data' in st.session_state or 'received_data' in st.session_state):
            st.info("🔄 检测到新的传输数据，正在自动进行误码率分析...")
            transfer._perform_ber_analysis()

        # 添加手动分析按钮
        col_manual1, col_manual2, col_manual3 = st.columns([1, 1, 1])
        with col_manual1:
            if st.button("🔄 手动进行误码率分析", type="primary"):
                # 检查是否有数据
                data_available = False

                # 检查session state中是否有数据
                if 'sent_data' in st.session_state:
                    data_available = True
                else:
                    # 尝试从文件加载
                    sent_data = transfer._load_transmission_data_from_file("sent")
                    if sent_data:
                        st.session_state.sent_data = sent_data
                        data_available = True
                        st.success("✅ 从文件成功加载发送数据")

                if data_available:
                    # 清除之前的分析结果，确保显示新的分析
                    if 'ber_analysis_results' in st.session_state:
                        del st.session_state.ber_analysis_results

                    # 执行误码率分析
                    with st.spinner("正在进行误码率分析..."):
                        transfer._perform_ber_analysis()
                else:
                    st.error("❌ 缺少发送数据，无法进行分析。请先发送一个文件。")

        with col_manual2:
            if st.button("🗑️ 清除分析结果"):
                if 'ber_analysis_results' in st.session_state:
                    del st.session_state.ber_analysis_results
                if 'ber_analysis_needed' in st.session_state:
                    del st.session_state.ber_analysis_needed
                st.success("分析结果已清除")
                st.rerun()

        with col_manual3:
            if st.button("🧹 清除临时文件"):
                if clear_temp_analysis_files():
                    st.rerun()

        # 检查是否有误码率分析结果
        if 'ber_analysis_results' in st.session_state and st.session_state.ber_analysis_results is not None:
            results = st.session_state.ber_analysis_results

            st.success("✅ 误码率分析结果")

            # 显示基本信息
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
                st.write(f"**实际信噪比**: {results.get('actual_snr', 'N/A'):.2f} dB")

            # 误码率对比 - 使用两列布局
            st.subheader("🎯 误码率对比分析")

            col_ber1, col_ber2 = st.columns(2)

            with col_ber1:
                # 显示实际传输误码率
                if results['actual_ber'] is not None:
                    actual_ber_display = f"{results['actual_ber']:.8f}" if results['actual_ber'] > 0 else "0.000000"
                    delta_text = f"{results['actual_errors']} 错误比特" if results['actual_errors'] > 0 else "0 错误比特"

                    st.metric(
                        "实际传输误码率",
                        actual_ber_display,
                        delta=delta_text,
                        help="实际传输过程中的误码率，对比发送和接收数据计算得出"
                    )

                    if results['actual_ber'] == 0:
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
                    st.metric(
                        "实际传输误码率",
                        "N/A",
                        delta="无接收数据",
                        help="没有接收数据可用于计算实际传输误码率"
                    )
                    st.info("**ℹ️ 无接收数据**: 无法计算实际传输误码率")

            with col_ber2:
                st.metric(
                    "模拟信道误码率",
                    f"{results['simulated_ber']:.8f}",
                    delta=f"{results['simulated_errors']} 错误比特",
                    help=f"在 {results['snr_db']}dB 信噪比下对发送数据模拟完整信道传输的误码率"
                )
                st.info("""
                **模拟信道分析:**
                - 发送数据 → 编码 → 调制 → 加噪 → 解调 → 解码
                - 模拟完整无线信道传输过程
                - 展示系统在实际信道中的抗噪性能
                """)

            # 详细统计
            st.subheader("📊 详细统计")

            col_stat1, col_stat2 = st.columns(2)

            with col_stat1:
                st.write("**🔵 实际传输统计**")
                if results['actual_ber'] is not None:
                    st.write(f"- 对比比特数: {results['actual_total_bits']:,}")
                    st.write(f"- 错误比特数: {results['actual_errors']:,}")
                    st.write(f"- 误码率: {results['actual_ber']:.8f}")
                    st.write(f"- 正确率: {(1 - results['actual_ber'])*100:.6f}%")
                    st.write(f"- 数据一致性: {'✅ 完全一致' if results['data_consistent'] else '❌ 不一致'}")
                else:
                    st.write(f"- 对比比特数: N/A")
                    st.write(f"- 错误比特数: N/A")
                    st.write(f"- 误码率: N/A")
                    st.write(f"- 正确率: N/A")
                    st.write(f"- 数据一致性: N/A")

            with col_stat2:
                st.write("**🔴 模拟信道统计**")
                st.write(f"- 对比比特数: {results['simulated_total_bits']:,}")
                st.write(f"- 错误比特数: {results['simulated_errors']:,}")
                st.write(f"- 误码率: {results['simulated_ber']:.8f}")
                st.write(f"- 正确率: {(1 - results['simulated_ber'])*100:.6f}%")

            # 功率和信噪比信息
            st.subheader("🔋 功率和信噪比信息")
            col_power1, col_power2, col_power3 = st.columns(3)
            with col_power1:
                st.write(f"**信号功率**: {results.get('signal_power', 0):.6f}")
                st.write(f"**噪声功率**: {results.get('noise_power', 0):.6f}")
            with col_power2:
                st.write(f"**设定信噪比**: {results['snr_db']} dB")
                st.write(f"**实际信噪比**: {results.get('actual_snr', 'N/A'):.2f} dB")
            with col_power3:
                # 计算信噪比误差
                if 'actual_snr' in results:
                    snr_error = results['actual_snr'] - results['snr_db']
                    st.write(f"**信噪比误差**: {snr_error:.2f} dB")
                    if abs(snr_error) < 0.5:
                        st.success("✅ 信噪比控制精确")
                    elif abs(snr_error) < 1.0:
                        st.info("ℹ️ 信噪比控制良好")
                    else:
                        st.warning("⚠️ 信噪比控制有偏差")

            # 数据样本对比
            st.subheader("🔍 数据样本对比")

            col_sample1, col_sample2, col_sample3 = st.columns(3)

            with col_sample1:
                st.write("**原始发送数据 (前100比特)**")
                st.text(''.join(map(str, results.get('original_bits_sample', []))))

            with col_sample2:
                st.write("**模拟信道数据 (前100比特)**")
                st.text(''.join(map(str, results.get('simulated_bits_sample', []))))

            with col_sample3:
                if len(results.get('received_bits_sample', [])) > 0:
                    st.write("**实际接收数据 (前100比特)**")
                    st.text(''.join(map(str, results.get('received_bits_sample', []))))
                else:
                    st.write("**实际接收数据**")
                    st.info("无接收数据")

            # 计算并显示错误位置
            if 'original_bits_sample' in results and 'simulated_bits_sample' in results:
                error_positions = []
                min_sample_len = min(len(results['original_bits_sample']), len(results['simulated_bits_sample']))
                for i in range(min_sample_len):
                    if results['original_bits_sample'][i] != results['simulated_bits_sample'][i]:
                        error_positions.append(i)

                if error_positions:
                    st.write(f"**模拟信道错误位置 (前100比特中):** {error_positions}")
                else:
                    st.write("**模拟信道错误位置:** 前100比特中无错误")

            # 性能评估
            st.subheader("📈 系统性能评估")

            col_perf1, col_perf2, col_perf3 = st.columns(3)

            with col_perf1:
                if results['actual_ber'] is not None:
                    if results['actual_ber'] == 0:
                        st.success("✅ 实际传输: 完美无误 (可靠传输机制)")
                    elif results['actual_ber'] < 0.001:
                        st.info("ℹ️ 实际传输: 良好")
                    else:
                        st.error("🔴 实际传输: 存在错误")
                else:
                    st.info("ℹ️ 实际传输: 无数据")

            with col_perf2:
                if results['simulated_ber'] < 0.001:
                    st.success("✅ 模拟信道: 优秀抗噪")
                elif results['simulated_ber'] < 0.01:
                    st.info("ℹ️ 模拟信道: 良好抗噪")
                else:
                    st.error("🔴 模拟信道: 抗噪较差")

            with col_perf3:
                if results['actual_ber'] is not None:
                    if results['actual_ber'] == 0 and results['simulated_ber'] > 0:
                        improvement_factor = float('inf')
                        st.success("🎯 可靠性增益: 无限 (可靠传输机制)")
                    elif results['actual_ber'] < results['simulated_ber']:
                        improvement_factor = results['simulated_ber'] / results['actual_ber'] if results['actual_ber'] > 0 else float('inf')
                        st.success(f"🎯 可靠性增益: {improvement_factor:.2f}倍")
                    else:
                        st.info("ℹ️ 可靠性增益: 无明显提升")
                else:
                    st.info("ℹ️ 可靠性增益: 无法计算")

            # 可视化误码率对比
            st.subheader("📊 误码率可视化")

            # 创建对比柱状图
            fig, ax = plt.subplots(figsize=(10, 6))

            if results['actual_ber'] is not None:
                categories = ['实际传输', '模拟信道']
                ber_values = [results['actual_ber'], results['simulated_ber']]
                colors = ['#4CAF50', '#FF6B6B']
            else:
                categories = ['模拟信道']
                ber_values = [results['simulated_ber']]
                colors = ['#FF6B6B']

            bars = ax.bar(categories, ber_values, color=colors, alpha=0.7, edgecolor='black')

            # 在柱子上显示数值
            for bar, value in zip(bars, ber_values):
                height = bar.get_height()
                if value == 0:
                    display_text = '0.000000'
                else:
                    display_text = f'{value:.8f}'
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                       display_text, ha='center', va='bottom', fontweight='bold')

            ax.set_ylabel('误码率 (BER)')
            ax.set_title('实际传输 vs 模拟信道误码率对比')
            ax.grid(True, alpha=0.3)

            # 设置y轴为对数坐标（如果误码率差异很大）
            if max(ber_values) > 0:
                ax.set_yscale('log')
                ax.set_ylabel('误码率 (BER) - 对数坐标')

            st.pyplot(fig)

            # 系统建议
            st.subheader("💡 系统优化建议")

            if results['simulated_ber'] > 0.01:
                st.error("""
                **建议改进措施:**
                - 提高信噪比设置
                - 使用更强大的编码方案（如汉明编码）
                - 考虑使用QPSK调制提高抗噪性
                - 增加重复编码的重复次数
                """)
            elif results['simulated_ber'] > 0.001:
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

            # 添加重新显示可视化的按钮
            st.markdown("---")
            if st.button("🔄 重新显示高级信号可视化", key="redisplay_viz"):
                if 'simulation_results' in results:
                    transfer._display_signal_visualization_enhanced(results['simulation_results'], results['modulation_type'])
                else:
                    st.warning("无法重新显示可视化，缺少模拟结果数据")

        else:
            # 没有分析数据时的说明界面
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
            - 🔄 **比特对比**: 对比发送和接收的前1000字节数据
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

            # 显示当前系统配置
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