"""调制解调与信道模拟模块（纯逻辑，不依赖 UI）。"""

from typing import List

import numpy as np

from .coding import decode_data, encode_data


class ModemSystem:
    """数字调制解调与信道模拟核心。"""

    def __init__(self) -> None:
        self.modulation_type = "BPSK"
        self.coding_rate = 1 / 2
        self.snr_db = 10
        self.bit_duration = 5

    # ---------- 调制 / 解调 ----------
    def modulate(self, bits, modulation_type: str = "BPSK") -> np.ndarray:
        """比特流 -> 基带符号序列。"""
        if modulation_type == "BPSK":
            symbols = np.where(bits, -1, 1)
            return np.repeat(symbols, self.bit_duration)
        if modulation_type == "QPSK":
            modulated = np.array([], dtype=complex)
            for i in range(0, len(bits) - 1, 2):
                dibit = (bits[i], bits[i + 1])
                if dibit == (0, 0):
                    symbol = complex(1, 1)
                elif dibit == (0, 1):
                    symbol = complex(-1, 1)
                elif dibit == (1, 0):
                    symbol = complex(-1, -1)
                else:
                    symbol = complex(1, -1)
                symbol /= np.sqrt(2)
                modulated = np.append(modulated, [symbol] * self.bit_duration)
            return modulated
        return np.array([])

    def demodulate(self, received_signal, modulation_type: str = "BPSK") -> List[int]:
        """基带符号序列 -> 比特流。"""
        bits: List[int] = []
        if modulation_type == "BPSK":
            for i in range(0, len(received_signal), self.bit_duration):
                end_idx = min(i + self.bit_duration, len(received_signal))
                symbol_avg = np.mean(received_signal[i:end_idx])
                bits.append(1 if symbol_avg < 0 else 0)
        elif modulation_type == "QPSK":
            for i in range(0, len(received_signal), self.bit_duration):
                end_idx = min(i + self.bit_duration, len(received_signal))
                symbol_avg = np.mean(received_signal[i:end_idx])
                real_part = np.real(symbol_avg)
                imag_part = np.imag(symbol_avg)
                if real_part >= 0 and imag_part >= 0:
                    bits.extend([0, 0])
                elif real_part < 0 and imag_part >= 0:
                    bits.extend([0, 1])
                elif real_part < 0 and imag_part < 0:
                    bits.extend([1, 0])
                else:
                    bits.extend([1, 1])
        return bits

    # ---------- 信道 ----------
    def add_noise(self, signal_data, snr_db) -> np.ndarray:
        """按给定信噪比（dB）叠加高斯白噪声。"""
        signal_power = self._signal_power(signal_data)
        noise_power = signal_power / (10 ** (snr_db / 10.0))

        if np.iscomplexobj(signal_data):
            noise = np.sqrt(noise_power / 2) * (
                np.random.randn(len(signal_data)) + 1j * np.random.randn(len(signal_data))
            )
        else:
            noise = np.sqrt(noise_power) * np.random.randn(len(signal_data))

        return signal_data + noise

    @staticmethod
    def calculate_ber(original_bits, received_bits) -> float:
        """计算误码率（BER）。"""
        min_len = min(len(original_bits), len(received_bits))
        if min_len == 0:
            return 0.0
        errors = int(np.sum(np.asarray(original_bits[:min_len]) != np.asarray(received_bits[:min_len])))
        return errors / min_len

    # ---------- 完整信道模拟 ----------
    def simulate_complete_channel(
        self,
        original_bits,
        modulation_type: str,
        coding_scheme: str,
        snr_db,
        seed: int = 42,
    ) -> dict:
        """完整信道模拟：编码 -> 调制 -> 加噪 -> 解调 -> 解码。

        返回结果字典，其中 ``steps`` 为供 UI 展示的过程说明。
        """
        np.random.seed(seed)

        steps: List[str] = []

        encoded_bits = encode_data(original_bits, coding_scheme)
        steps.append("**步骤1: 编码**")
        steps.append(f"- 原始比特数: {len(original_bits)}")
        steps.append(f"- 编码后比特数: {len(encoded_bits)}")
        steps.append(f"- 编码效率: {len(original_bits) / len(encoded_bits):.3f}")

        modulated_signal = self.modulate(encoded_bits, modulation_type)
        steps.append("**步骤2: 调制**")
        steps.append(f"- 调制后信号长度: {len(modulated_signal)}")
        steps.append(f"- 调制类型: {modulation_type}")

        signal_power = self._signal_power(modulated_signal)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        steps.append("**步骤3: 添加噪声**")
        steps.append(f"- 信号功率: {signal_power:.6f}")
        steps.append(f"- 信噪比: {snr_db} dB (线性: {snr_linear:.3f})")
        steps.append(f"- 理论噪声功率: {noise_power:.6f}")

        noisy_signal = self.add_noise(modulated_signal, snr_db)
        actual_noise_power = self._signal_power(noisy_signal - modulated_signal)
        actual_snr = 10 * np.log10(signal_power / actual_noise_power) if actual_noise_power > 0 else float("inf")
        steps.append(f"- 实际噪声功率: {actual_noise_power:.6f}")
        steps.append(f"- 实际信噪比: {actual_snr:.2f} dB")

        demodulated_bits = self.demodulate(noisy_signal, modulation_type)
        steps.append("**步骤4: 解调**")
        steps.append(f"- 解调后比特数: {len(demodulated_bits)}")

        decoded_bits = decode_data(demodulated_bits, coding_scheme)
        steps.append("**步骤5: 解码**")
        steps.append(f"- 解码后比特数: {len(decoded_bits)}")

        min_len = min(len(original_bits), len(decoded_bits))
        errors = int(np.sum(np.asarray(original_bits[:min_len]) != np.asarray(decoded_bits[:min_len]))) if min_len > 0 else 0
        simulated_ber = errors / min_len if min_len > 0 else 0.0

        steps.append("**步骤6: 误码率计算**")
        steps.append(f"- 对比比特数: {min_len}")
        steps.append(f"- 错误比特数: {errors}")
        steps.append(f"- 模拟误码率: {simulated_ber:.6f}")

        return {
            "encoded_bits": encoded_bits,
            "modulated_signal": modulated_signal,
            "noisy_signal": noisy_signal,
            "demodulated_bits": demodulated_bits,
            "decoded_bits": decoded_bits,
            "simulated_ber": simulated_ber,
            "actual_snr": actual_snr,
            "signal_power": signal_power,
            "noise_power": actual_noise_power,
            "errors": errors,
            "compared_bits": min_len,
            "steps": steps,
        }

    @staticmethod
    def _signal_power(signal) -> float:
        """计算信号平均功率。"""
        if np.iscomplexobj(signal):
            return float(np.mean(np.real(signal) ** 2 + np.imag(signal) ** 2))
        return float(np.mean(signal ** 2))
