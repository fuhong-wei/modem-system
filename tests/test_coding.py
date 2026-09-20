"""信道编码单元测试。"""

import pytest

from src.coding import (
    decode_data,
    encode_data,
    hamming_decode,
    hamming_encode,
    repetition_decode,
    repetition_encode,
)


class TestRepetition:
    def test_encode(self):
        assert repetition_encode([1, 0, 1]) == [1, 1, 1, 0, 0, 0, 1, 1, 1]

    def test_roundtrip(self):
        bits = [1, 0, 1, 1, 0, 0, 1, 0]
        assert repetition_decode(repetition_encode(bits)) == bits

    def test_corrects_single_error(self):
        encoded = repetition_encode([1, 0, 1])
        encoded[0] = 0  # 翻转第 1 个比特
        assert repetition_decode(encoded) == [1, 0, 1]


class TestHamming:
    def test_roundtrip(self):
        bits = [1, 0, 1, 1, 0, 1, 0, 0]
        assert hamming_decode(hamming_encode(bits)) == bits

    @pytest.mark.parametrize("data", [[0, 0, 0, 0], [1, 1, 1, 1], [1, 0, 1, 1], [0, 1, 0, 1]])
    def test_corrects_single_bit_error_at_every_position(self, data):
        encoded = hamming_encode(data)
        for pos in range(7):
            corrupted = list(encoded)
            corrupted[pos] ^= 1
            assert hamming_decode(corrupted) == data, f"位置 {pos} 纠错失败"


class TestDispatch:
    def test_none_is_passthrough(self):
        bits = [1, 0, 1, 0]
        assert encode_data(bits, "none") == bits
        assert decode_data(bits, "none") == bits

    def test_unknown_scheme_is_passthrough(self):
        bits = [1, 0, 1, 0]
        assert encode_data(bits, "unknown") == bits
        assert decode_data(bits, "unknown") == bits
