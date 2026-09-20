"""调制解调与信道模拟单元测试。"""

import numpy as np
import pytest

from src.modulation import ModemSystem


@pytest.fixture
def modem() -> ModemSystem:
    return ModemSystem()


class TestModulateDemodulate:
    @pytest.mark.parametrize("mt", ["BPSK", "QPSK"])
    def test_roundtrip(self, modem, mt):
        bits = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0]
        assert modem.demodulate(modem.modulate(bits, mt), mt) == bits

    def test_bpsk_symbols(self, modem):
        signal = modem.modulate([0, 1], "BPSK")
        assert np.all(signal[:modem.bit_duration] == 1)
        assert np.all(signal[modem.bit_duration:] == -1)


class TestAddNoise:
    def test_high_snr_has_no_error(self, modem):
        bits = [1, 0, 1, 1, 0, 0, 1, 0]
        np.random.seed(0)
        noisy = modem.add_noise(modem.modulate(bits, "BPSK"), 30)
        assert modem.calculate_ber(bits, modem.demodulate(noisy, "BPSK")) == 0.0

    def test_low_snr_has_errors(self, modem):
        bits = [1, 0, 1, 1] * 50  # 200 比特，保证低信噪比下必然出现误码
        np.random.seed(1)
        noisy = modem.add_noise(modem.modulate(bits, "BPSK"), -10)
        assert modem.calculate_ber(bits, modem.demodulate(noisy, "BPSK")) > 0.0


class TestCalculateBER:
    def test_perfect_match(self, modem):
        assert modem.calculate_ber([1, 0, 1], [1, 0, 1]) == 0.0

    def test_all_different(self, modem):
        assert modem.calculate_ber([1, 1, 1, 1], [0, 0, 0, 0]) == 1.0

    def test_empty(self, modem):
        assert modem.calculate_ber([], []) == 0.0


class TestSimulateCompleteChannel:
    def test_returns_expected_keys(self, modem):
        bits = [1, 0, 1, 1, 0, 0, 1, 0]
        result = modem.simulate_complete_channel(bits, "BPSK", "repetition", 10)
        for key in ("encoded_bits", "modulated_signal", "noisy_signal", "decoded_bits", "simulated_ber", "steps"):
            assert key in result

    def test_reproducible_with_seed(self, modem):
        bits = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0]
        r1 = modem.simulate_complete_channel(bits, "BPSK", "repetition", 5, seed=42)
        r2 = modem.simulate_complete_channel(bits, "BPSK", "repetition", 5, seed=42)
        assert r1["simulated_ber"] == r2["simulated_ber"]
