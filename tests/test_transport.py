"""可靠 UDP 传输的纯逻辑单元测试。"""

from src.transport import ReliableUDPTransfer


class TestBitByteConversion:
    def test_roundtrip(self):
        transfer = ReliableUDPTransfer()
        data = b"Hello, modem!"
        assert transfer.bits_to_bytes(transfer.bytes_to_bits(data)) == data

    def test_known_bits(self):
        transfer = ReliableUDPTransfer()
        assert transfer.bytes_to_bits(b"\x01") == [0, 0, 0, 0, 0, 0, 0, 1]

    def test_empty(self):
        transfer = ReliableUDPTransfer()
        assert transfer.bits_to_bytes([]) == b""


class TestSaveTransmissionData:
    def test_record_structure(self, tmp_path, monkeypatch):
        transfer = ReliableUDPTransfer()
        monkeypatch.chdir(tmp_path)  # 重定向临时目录，避免污染工作目录

        messages = []
        transfer.on_message = lambda level, text: messages.append((level, text))

        record = transfer.save_transmission_data(b"abcdefgh", "test.txt", "sent", "BPSK", "重复编码", 10)

        assert record["filename"] == "test.txt"
        assert record["file_size"] == 8
        assert record["modulation_type"] == "BPSK"
        assert record["coding_scheme"] == "重复编码"
        assert len(record["original_bits"]) == 8 * 8
        assert any(level == "success" for level, _ in messages)
