"""信道编码模块：重复编码与汉明编码（纯函数，无副作用）。"""

from typing import List


def repetition_encode(bits: List[int], repeat: int = 3) -> List[int]:
    """重复编码 (1, n)：每个比特重复 n 次。"""
    encoded: List[int] = []
    for bit in bits:
        encoded.extend([bit] * repeat)
    return encoded


def repetition_decode(bits: List[int], repeat: int = 3) -> List[int]:
    """重复编码解码：多数表决。"""
    decoded: List[int] = []
    for i in range(0, len(bits), repeat):
        chunk = bits[i:i + repeat]
        if chunk:
            decoded.append(1 if sum(chunk) * 2 > len(chunk) else 0)
    return decoded


def hamming_encode(bits: List[int]) -> List[int]:
    """汉明编码 (7,4)：4 数据位 + 3 校验位。"""
    encoded: List[int] = []
    for i in range(0, len(bits), 4):
        chunk = bits[i:i + 4]
        if len(chunk) < 4:
            chunk = chunk + [0] * (4 - len(chunk))
        d1, d2, d3, d4 = chunk[0], chunk[1], chunk[2], chunk[3]
        p1 = d1 ^ d2 ^ d4
        p2 = d1 ^ d3 ^ d4
        p3 = d2 ^ d3 ^ d4
        encoded.extend([d1, d2, d3, d4, p1, p2, p3])
    return encoded


def hamming_decode(bits: List[int]) -> List[int]:
    """汉明解码：计算校验子并纠正单比特错误，返回前 4 位数据。

    码字布局为 ``[d1 d2 d3 d4 p1 p2 p3]``（系统码，数据位在前）。校验子
    ``(s1, s2, s3)`` 由三个校验方程与接收到的校验位异或得到，再通过查找表
    定位出错的比特位置。
    """
    # 校验子 (s1, s2, s3) -> 码字中出错比特的下标
    syndrome_to_index = {
        (1, 1, 0): 0,  # d1
        (1, 0, 1): 1,  # d2
        (0, 1, 1): 2,  # d3
        (1, 1, 1): 3,  # d4
        (1, 0, 0): 4,  # p1
        (0, 1, 0): 5,  # p2
        (0, 0, 1): 6,  # p3
    }

    decoded: List[int] = []
    for i in range(0, len(bits), 7):
        chunk = bits[i:i + 7]
        if len(chunk) == 7:
            d1, d2, d3, d4, p1, p2, p3 = chunk
            s1 = p1 ^ d1 ^ d2 ^ d4
            s2 = p2 ^ d1 ^ d3 ^ d4
            s3 = p3 ^ d2 ^ d3 ^ d4
            error_index = syndrome_to_index.get((s1, s2, s3))
            if error_index is not None:
                chunk[error_index] = 1 - chunk[error_index]
            decoded.extend(chunk[:4])
    return decoded


def encode_data(data_bits: List[int], coding_scheme: str = "repetition") -> List[int]:
    """按编码方案编码数据。方案：repetition / hamming / none。"""
    if coding_scheme == "repetition":
        return repetition_encode(data_bits)
    if coding_scheme == "hamming":
        return hamming_encode(data_bits)
    return data_bits


def decode_data(encoded_bits: List[int], coding_scheme: str = "repetition") -> List[int]:
    """按编码方案解码数据。"""
    if coding_scheme == "repetition":
        return repetition_decode(encoded_bits)
    if coding_scheme == "hamming":
        return hamming_decode(encoded_bits)
    return encoded_bits
