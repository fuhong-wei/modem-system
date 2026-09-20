"""可靠 UDP 传输模块：分块、滑动窗口、超时重传与 ACK 确认（纯逻辑，不依赖 UI）。

消息、进度、状态通过可选回调暴露给上层（如 Streamlit），核心类本身可独立测试。
"""

import glob
import logging
import os
import socket
import struct
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .modulation import ModemSystem

logger = logging.getLogger(__name__)

MessageCallback = Callable[[str, str], None]
ProgressCallback = Callable[[int, int], None]
StatusCallback = Callable[[str], None]


class ReliableUDPTransfer:
    """基于 UDP 的可靠文件传输：分块、滑动窗口、超时重传、ACK 确认。"""

    def __init__(
        self,
        on_message: Optional[MessageCallback] = None,
        on_progress: Optional[ProgressCallback] = None,
        on_status: Optional[StatusCallback] = None,
    ) -> None:
        self.chunk_size = 4096
        self.timeout = 3.0
        self.max_retries = 5
        self.window_size = 8
        self.modem = ModemSystem()
        self.ack_timeout = 0.5

        self.on_message = on_message or (lambda level, text: None)
        self.on_progress = on_progress or (lambda done, total: None)
        self.on_status = on_status or (lambda text: None)

        self._sent_time: Dict[int, float] = {}
        self._retry_count: Dict[int, int] = {}

    # ---------- 字节 <-> 比特 ----------
    @staticmethod
    def bytes_to_bits(data: bytes) -> List[int]:
        """字节数据 -> 比特列表。"""
        data_array = np.frombuffer(data, dtype=np.uint8)
        return np.unpackbits(data_array).tolist()

    @staticmethod
    def bits_to_bytes(bits: List[int]) -> bytes:
        """比特列表 -> 字节数据。"""
        padded_bits = bits + [0] * ((8 - len(bits) % 8) % 8)
        bit_array = np.array(padded_bits, dtype=np.uint8)
        return np.packbits(bit_array).tobytes()

    def _reset_state(self) -> None:
        """清理发送端的协议状态。"""
        self._sent_time.clear()
        self._retry_count.clear()

    # ---------- 发送 ----------
    def send_file(
        self,
        target_ip: str,
        target_port: int,
        file_data: bytes,
        filename: str,
        modulation_type: str,
        coding_scheme: str,
        snr_db,
    ):
        """发送文件，返回用于误码率分析的数据记录；失败返回 False。"""
        try:
            file_size = len(file_data)
            total_chunks = (file_size + self.chunk_size - 1) // self.chunk_size

            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(self.ack_timeout)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)

            file_info = (
                f"FILE_INFO|{filename}|{file_size}|{total_chunks}"
                f"|{modulation_type}|{coding_scheme}|{snr_db}"
            ).encode()

            established = False
            for retry in range(self.max_retries):
                try:
                    sock.sendto(file_info, (target_ip, target_port))
                    data, _addr = sock.recvfrom(1024)
                    if data == b"ACK_FILE_INFO":
                        self.on_message("success", "接收端已确认文件信息")
                        established = True
                        break
                except socket.timeout:
                    if retry == self.max_retries - 1:
                        self.on_message("error", "无法建立连接：接收端未响应")
                        sock.close()
                        return False
                    self.on_message("warning", f"文件信息确认超时，重试 {retry + 1}/{self.max_retries}")

            if not established:
                sock.close()
                return False

            self._reset_state()
            sent_chunks = 0
            acked_chunks: set = set()
            window_start = 0
            last_ack_time = time.time()
            start_time = time.time()

            while sent_chunks < total_chunks or len(acked_chunks) < total_chunks:
                current_time = time.time()

                # 发送窗口内的数据包
                while sent_chunks < total_chunks and sent_chunks < window_start + self.window_size:
                    chunk_idx = sent_chunks
                    start = chunk_idx * self.chunk_size
                    end = min((chunk_idx + 1) * self.chunk_size, file_size)
                    chunk_data = file_data[start:end]

                    packet_header = struct.pack("!II", chunk_idx, len(chunk_data))
                    packet = packet_header + chunk_data
                    try:
                        sock.sendto(packet, (target_ip, target_port))
                        self._sent_time[chunk_idx] = current_time
                        sent_chunks += 1

                        if sent_chunks % 10 == 0 or sent_chunks == total_chunks:
                            self.on_progress(sent_chunks, total_chunks)
                            elapsed = current_time - start_time
                            speed = (sent_chunks * self.chunk_size / 1024 / 1024) / elapsed if elapsed > 0 else 0
                            self.on_status(f"发送进度: {sent_chunks}/{total_chunks} | 速度: {speed:.2f} MB/s")
                    except Exception as e:
                        self.on_message("error", f"发送数据包 {chunk_idx} 失败: {e}")
                        break

                # 接收 ACK
                try:
                    sock.settimeout(0.1)
                    data, _addr = sock.recvfrom(1024)
                    if data.startswith(b"ACK"):
                        ack_idx = int(data.split(b"|")[1])
                        acked_chunks.add(ack_idx)
                        last_ack_time = time.time()

                        while window_start in acked_chunks:
                            window_start += 1

                        self.on_progress(len(acked_chunks), total_chunks)
                        elapsed = time.time() - start_time
                        speed = (len(acked_chunks) * self.chunk_size / 1024 / 1024) / elapsed if elapsed > 0 else 0
                        self.on_status(f"确认进度: {len(acked_chunks)}/{total_chunks} | 速度: {speed:.2f} MB/s")
                except socket.timeout:
                    pass
                except Exception as e:
                    self.on_message("warning", f"接收ACK时出错: {e}")

                # 超时重传
                current_time = time.time()
                for chunk_idx in range(window_start, min(window_start + self.window_size, total_chunks)):
                    if chunk_idx not in acked_chunks and current_time - self._sent_time.get(chunk_idx, 0) > self.timeout:
                        if self._retry_count.get(chunk_idx, 0) < self.max_retries:
                            start = chunk_idx * self.chunk_size
                            end = min((chunk_idx + 1) * self.chunk_size, file_size)
                            chunk_data = file_data[start:end]
                            packet_header = struct.pack("!II", chunk_idx, len(chunk_data))
                            packet = packet_header + chunk_data
                            try:
                                sock.sendto(packet, (target_ip, target_port))
                                self._sent_time[chunk_idx] = current_time
                                self._retry_count[chunk_idx] = self._retry_count.get(chunk_idx, 0) + 1
                                self.on_message("warning", f"重传数据包 {chunk_idx} (尝试 {self._retry_count[chunk_idx]})")
                            except Exception as e:
                                self.on_message("error", f"重传数据包 {chunk_idx} 失败: {e}")
                        else:
                            self.on_message("error", f"数据包 {chunk_idx} 达到最大重传次数，传输失败")
                            sock.close()
                            return False

                if current_time - last_ack_time > self.timeout * 2:
                    self.on_message("error", "传输超时，连接可能已断开")
                    sock.close()
                    return False

                time.sleep(0.001)

            sock.sendto(b"END_OF_TRANSMISSION", (target_ip, target_port))

            total_time = time.time() - start_time
            total_speed = (file_size / 1024 / 1024) / total_time if total_time > 0 else 0
            self.on_message("success", f"文件发送完成! 总时间: {total_time:.2f}秒, 平均速度: {total_speed:.2f} MB/s")
            sock.close()

            return self.save_transmission_data(file_data, filename, "sent", modulation_type, coding_scheme, snr_db)

        except Exception as e:
            self.on_message("error", f"发送失败: {str(e)}")
            return False

    # ---------- 传输数据持久化 ----------
    def save_transmission_data(
        self,
        file_data: bytes,
        filename: str,
        direction: str,
        modulation_type: Optional[str] = None,
        coding_scheme: Optional[str] = None,
        snr_db: Optional[float] = None,
    ) -> Dict[str, Any]:
        """保存传输数据（前 1000 字节），返回数据记录，并备份到临时文件。"""
        modulation_type = modulation_type or self.modem.modulation_type
        coding_scheme = coding_scheme or "重复编码"
        snr_db = snr_db if snr_db is not None else self.modem.snr_db

        sample_data = file_data[:1000] if len(file_data) > 1000 else file_data
        bits = self.bytes_to_bits(sample_data)

        record = {
            "original_bits": bits,
            "filename": filename,
            "file_size": len(file_data),
            "timestamp": time.time(),
            "modulation_type": modulation_type,
            "coding_scheme": coding_scheme,
            "snr_db": snr_db,
        }

        try:
            os.makedirs("./temp_analysis", exist_ok=True)
            temp_file = f"./temp_analysis/{direction}_{filename}_{int(time.time())}.dat"
            with open(temp_file, "wb") as f:
                f.write(sample_data)
        except Exception as e:
            self.on_message("error", f"保存传输数据错误: {e}")

        self.on_message("success", f"✅ {direction}数据已保存用于误码率分析 (已备份到文件)")
        return record

    def load_transmission_data_from_file(self, direction: str) -> Optional[Dict[str, Any]]:
        """从最新临时文件加载传输数据。"""
        try:
            if not os.path.exists("./temp_analysis"):
                return None

            files = glob.glob(f"./temp_analysis/{direction}_*.dat")
            if not files:
                return None

            latest_file = max(files, key=os.path.getmtime)
            with open(latest_file, "rb") as f:
                file_data = f.read()

            filename = os.path.basename(latest_file).split("_", 2)[2].rsplit("_", 1)[0]
            bits = self.bytes_to_bits(file_data)

            return {
                "original_bits": bits,
                "filename": filename,
                "file_size": len(file_data),
                "timestamp": os.path.getmtime(latest_file),
                "modulation_type": self.modem.modulation_type,
                "coding_scheme": "重复编码",
                "snr_db": self.modem.snr_db,
            }
        except Exception as e:
            self.on_message("error", f"从文件加载数据错误: {e}")
            return None

    # ---------- 接收 ----------
    def start_receiver(self, listen_port: int, save_dir: str):
        """启动接收端，接收文件并返回用于误码率分析的数据记录。"""
        try:
            os.makedirs(save_dir, exist_ok=True)

            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
            sock.bind(("0.0.0.0", listen_port))
            sock.settimeout(1.0)

            self.on_message("success", f"开始在端口 {listen_port} 监听...")

            received_data: Dict[int, bytes] = {}
            total_chunks = 0
            filename = ""
            start_time = time.time()
            last_packet_time = time.time()

            receiver_modulation_type = "BPSK"
            receiver_coding_scheme = "重复编码"
            receiver_snr_db = 10.0

            self.on_progress(0, 1)
            self._reset_state()

            while True:
                try:
                    data, addr = sock.recvfrom(self.chunk_size + 8)
                    last_packet_time = time.time()

                    if data.startswith(b"FILE_INFO"):
                        info_parts = data.decode().split("|")
                        filename = info_parts[1]
                        total_chunks = int(info_parts[3])

                        if len(info_parts) >= 7:
                            receiver_modulation_type = info_parts[4]
                            receiver_coding_scheme = info_parts[5]
                            receiver_snr_db = float(info_parts[6])
                            self.on_message(
                                "info",
                                f"📡 从发送端接收参数: 调制={receiver_modulation_type}, "
                                f"编码={receiver_coding_scheme}, 信噪比={receiver_snr_db}dB",
                            )

                        sock.sendto(b"ACK_FILE_INFO", addr)
                        self.on_status(f"开始接收文件: {filename}")
                        start_time = time.time()
                        continue

                    if data == b"END_OF_TRANSMISSION":
                        self.on_message("success", "传输完成确认收到")
                        break

                    if len(data) >= 8:
                        try:
                            seq_num, data_length = struct.unpack("!II", data[:8])
                            chunk_data = data[8:8 + data_length]

                            if seq_num not in received_data:
                                received_data[seq_num] = chunk_data
                                sock.sendto(f"ACK|{seq_num}".encode(), addr)

                            if total_chunks > 0:
                                current = len(received_data)
                                self.on_progress(current, total_chunks)
                                elapsed = time.time() - start_time
                                received_size = sum(len(d) for d in received_data.values())
                                speed = (received_size / 1024 / 1024) / elapsed if elapsed > 0 else 0
                                self.on_status(f"接收进度: {current}/{total_chunks} | 速度: {speed:.2f} MB/s")

                            if len(received_data) >= total_chunks > 0:
                                break

                        except struct.error as e:
                            self.on_message("warning", f"数据包解析错误: {e}")
                            continue

                except socket.timeout:
                    if time.time() - last_packet_time > self.timeout * 3:
                        self.on_message("warning", "接收超时，可能传输已结束")
                        break
                    continue

                except Exception as e:
                    self.on_message("error", f"接收错误: {str(e)}")
                    if len(received_data) >= total_chunks > 0:
                        break
                    continue

            sock.close()

            # 重组文件（即使不完整也保存）
            if not received_data:
                self.on_message("error", "未接收到有效数据")
                return None

            sorted_data: List[bytes] = []
            missing_chunks: List[int] = []
            for i in range(total_chunks):
                if i in received_data:
                    sorted_data.append(received_data[i])
                else:
                    missing_chunks.append(i)

            if missing_chunks:
                self.on_message("warning", f"缺失 {len(missing_chunks)} 个数据块: {missing_chunks}")

            file_data = b"".join(sorted_data)

            save_path = os.path.join(save_dir, f"received_{filename}")
            try:
                with open(save_path, "wb") as f:
                    f.write(file_data)
                self.on_message("success", f"文件保存成功: {save_path}")
            except Exception as save_error:
                self.on_message("error", f"文件保存失败: {save_error}")

            elapsed = time.time() - start_time
            file_size_mb = len(file_data) / 1024 / 1024
            speed = file_size_mb / elapsed if elapsed > 0 else 0
            completion_ratio = len(received_data) / total_chunks if total_chunks > 0 else 0

            if completion_ratio < 1.0:
                self.on_message("warning", f"文件接收不完整: {completion_ratio:.1%}")
            self.on_message(
                "info",
                f"传输统计: 大小 {file_size_mb:.2f} MB, 时间 {elapsed:.2f} 秒, "
                f"平均速度 {speed:.2f} MB/s, 完整度 {completion_ratio:.1%}",
            )

            return self.save_transmission_data(
                file_data, filename, "received",
                receiver_modulation_type, receiver_coding_scheme, receiver_snr_db,
            )

        except Exception as e:
            self.on_message("error", f"启动接收端失败: {str(e)}")
            return None
