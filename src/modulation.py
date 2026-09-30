"""调制解调与信道模拟模块（纯逻辑，不依赖 UI）。"""

from typing import List, Optional

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
            bit_array = np.asarray(bits, dtype=np.uint8)
            # QPSK 两比特一符号，丢掉落单的最后一个比特（与原逻辑 range(0, len-1, 2) 一致）
            bit_array = bit_array[: len(bit_array) // 2 * 2]
            if bit_array.size == 0:
                return np.array([], dtype=complex)
            dibits = bit_array.reshape(-1, 2)
            # 索引 = b0*2 + b1，顺序与原映射一致：00->1+1j, 01->-1+1j, 10->-1-1j, 11->1-1j
            idx = dibits[:, 0].astype(np.intp) * 2 + dibits[:, 1].astype(np.intp)
            table = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=complex) / np.sqrt(2)
            return np.repeat(table[idx], self.bit_duration)
        return np.array([])

    def demodulate(self, received_signal, modulation_type: str = "BPSK") -> List[int]:
        """基带符号序列 -> 比特流（向量化，避免逐符号 Python 循环）。"""
        raw = np.asarray(received_signal)
        if raw.size == 0:
            return []
        sig = raw.astype(complex) if np.iscomplexobj(raw) else raw.astype(float)

        d = self.bit_duration
        n_full = sig.size // d
        # 按 bit_duration 分组求均值；不足一组的尾部单独平均（与原循环行为一致）
        parts = []
        if n_full:
            parts.append(sig[: n_full * d].reshape(n_full, d).mean(axis=1))
        tail = sig[n_full * d:]
        if tail.size:
            parts.append(np.array([tail.mean()], dtype=sig.dtype))
        symbol_avgs = np.concatenate(parts) if parts else np.array([], dtype=sig.dtype)

        if modulation_type == "BPSK":
            return (np.real(symbol_avgs) < 0).astype(np.uint8).tolist()

        if modulation_type == "QPSK":
            i_neg = np.real(symbol_avgs) < 0
            q_neg = np.imag(symbol_avgs) < 0
            # 与原 if/elif 判决表等价：
            # (I>=0,Q>=0)->00  (I<0,Q>=0)->01  (I<0,Q<0)->10  (I>=0,Q<0)->11
            # 即 bit0 = (Q<0)，bit1 = (I<0) XOR (Q<0)
            out = np.empty(symbol_avgs.size * 2, dtype=np.uint8)
            out[0::2] = q_neg
            out[1::2] = i_neg ^ q_neg
            return out.tolist()

        return []

    # ---------- 信道 ----------
    def add_noise(self, signal_data, snr_db, rng=None) -> np.ndarray:
        """按给定信噪比（dB）叠加高斯白噪声。

        ``rng`` 为可选的 numpy Generator。不传则新建一个独立生成器，
        不再像 ``np.random.randn`` 那样污染全局随机状态。
        """
        generator = rng if rng is not None else np.random.default_rng()
        signal_power = self._signal_power(signal_data)
        noise_power = signal_power / (10 ** (snr_db / 10.0))
        n = len(signal_data)

        if np.iscomplexobj(signal_data):
            noise = np.sqrt(noise_power / 2) * (
                generator.standard_normal(n) + 1j * generator.standard_normal(n)
            )
        else:
            noise = np.sqrt(noise_power) * generator.standard_normal(n)

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
        seed: Optional[int] = 42,
    ) -> dict:
        """完整信道模拟：编码 -> 调制 -> 加噪 -> 解调 -> 解码。

        返回结果字典，其中 ``steps`` 为供 UI 展示的过程说明。

        ``seed`` 传 int 则结果可复现，传 None 则每次噪声都不同。
        用局部 Generator 而不是 ``np.random.seed``，避免污染全局随机状态。
        """
        rng = np.random.default_rng(seed)

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

        noisy_signal = self.add_noise(modulated_signal, snr_db, rng=rng)
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
