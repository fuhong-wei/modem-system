"""传输数据持久化 / 恢复的回归测试。"""

import os

import numpy as np
import pytest

from src.transport import ReliableUDPTransfer


@pytest.fixture
def transfer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # temp_analysis 建在临时目录里
    return ReliableUDPTransfer()


def test_bytes_to_bits_returns_ndarray(transfer):
    """必须返回 ndarray：返回 Python list 会让大文件爆内存。"""
    bits = transfer.bytes_to_bits(b"\x01")
    assert isinstance(bits, np.ndarray)
    assert np.array_equal(bits, [0, 0, 0, 0, 0, 0, 0, 1])


def test_bits_to_bytes_roundtrip(transfer):
    data = b"hello modem"
    assert transfer.bits_to_bytes(transfer.bytes_to_bits(data)) == data


@pytest.mark.parametrize("filename", ["a.txt", "my_file.bin", "报告_v2.pdf"])
def test_save_load_roundtrip_preserves_metadata(transfer, filename):
    """元数据必须通过 .json 边车原样恢复，不能退回瞎猜的默认值。"""
    transfer.save_transmission_data(
        b"payload-bytes", filename, "sent", "QPSK", "汉明编码", 25,
        duration=1.5, speed=3.5, retransmissions=7, retransmitted_chunks=3,
        total_chunks=10,
    )
    loaded = transfer.load_transmission_data_from_file("sent")

    assert loaded is not None
    assert loaded["filename"] == filename          # 历史 bug：会变成时间戳
    assert loaded["modulation_type"] == "QPSK"     # 历史 bug：永远是 BPSK
    assert loaded["coding_scheme"] == "汉明编码"    # 历史 bug：硬编码"重复编码"
    assert loaded["snr_db"] == 25                  # 历史 bug：永远是 10
    assert loaded["duration"] == pytest.approx(1.5)
    assert loaded["retransmissions"] == 7
    assert loaded["retransmitted_chunks"] == 3


def test_received_direction_uses_received_bits_key(transfer):
    """received 方向必须存 received_bits，不能叫 original_bits。"""
    record = transfer.save_transmission_data(
        b"abc", "x.bin", "received", "BPSK", "无编码", 10, completion_ratio=0.8,
    )
    assert "received_bits" in record
    assert "original_bits" not in record
    assert transfer.load_transmission_data_from_file("received")["completion_ratio"] == 0.8


def test_temp_analysis_is_pruned(transfer):
    """临时文件不能无限堆积。"""
    import glob
    import time

    for i in range(8):
        transfer.save_transmission_data(b"x" * 10, f"f{i}.bin", "sent")
        time.sleep(0.01)  # 保证 mtime 有区分度
    assert len(glob.glob("./temp_analysis/sent_*.dat")) <= 5
    # 每个 .dat 都应有配对的 .json
    for dat in glob.glob("./temp_analysis/sent_*.dat"):
        assert os.path.exists(dat[:-4] + ".json")
