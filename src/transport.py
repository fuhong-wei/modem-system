"""可靠 UDP 传输模块：分块、滑动窗口、超时重传与 ACK 确认（纯逻辑，不依赖 UI）。

消息、进度、状态通过可选回调暴露给上层（如 Streamlit），核心类本身可独立测试。
"""

import glob
import logging
import os
import socket
import struct
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

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
        self.sample_size = 64 * 1024  # 误码率分析的采样字节数；None 表示整个文件

        self.on_message = on_message or (lambda level, text: None)
        self.on_progress = on_progress or (lambda done, total: None)
        self.on_status = on_status or (lambda text: None)

        self._sent_time: Dict[int, float] = {}
        self._retry_count: Dict[int, int] = {}

        # 接收端生命周期控制（供 UI 停止监听 / 显示实时状态）
        self._receiver_stop = threading.Event()
        self._receiver_socket: Optional[socket.socket] = None
        self._receiver_alive = threading.Event()
        self._receiver_generation = 0
        self._receiver_thread_ident: Optional[int] = None

        # 接收线程运行状态（线程安全，供 UI 主线程读取）
        self.receiver_error: Optional[str] = None
        self.receiver_actual_port: Optional[int] = None
        self.last_received_record: Optional[Dict[str, Any]] = None

        # 后台接收线程的消息/进度/状态缓冲（避免后台线程直接调用 st.*）
        self._receiver_lock = threading.Lock()
        self._receiver_messages: List[Tuple[str, str]] = []
        self._receiver_progress: Tuple[int, int] = (0, 0)
        self._receiver_status = ""

    @property
    def receiver_alive(self) -> bool:
        """接收端是否正在监听（线程安全，由接收线程 + generation 兜底维护）。"""
        return self._receiver_alive.is_set()

    # ---------- 接收线程事件缓冲（后台线程写入，UI 主线程读取） ----------
    def _receiver_log(self, level: str, text: str) -> None:
        with self._receiver_lock:
            self._receiver_messages.append((level, text))

    def _receiver_update_progress(self, done: int, total: int) -> None:
        with self._receiver_lock:
            self._receiver_progress = (done, total)

    def _receiver_update_status(self, text: str) -> None:
        with self._receiver_lock:
            self._receiver_status = text

    def _notify(self, level: str, text: str) -> None:
        """统一消息出口：接收线程写入缓冲，其余线程直接回调（避免后台线程调 st.*）。"""
        if threading.get_ident() == self._receiver_thread_ident:
            self._receiver_log(level, text)
        else:
            self.on_message(level, text)

    def drain_receiver_events(self) -> Dict[str, Any]:
        """取出并清空接收线程缓冲的消息/进度/状态（UI 主线程调用）。"""
        with self._receiver_lock:
            messages = self._receiver_messages[:]
            self._receiver_messages.clear()
            progress = self._receiver_progress
            status = self._receiver_status
        return {"messages": messages, "progress": progress, "status": status}

    def set_last_received_record(self, record: Optional[Dict[str, Any]]) -> None:
        """接收线程把结果记录交给 UI 主线程（线程安全）。"""
        with self._receiver_lock:
            self.last_received_record = record

    def take_last_received_record(self) -> Optional[Dict[str, Any]]:
        with self._receiver_lock:
            record = self.last_received_record
            self.last_received_record = None
        return record

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
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)

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
                except ConnectionResetError:
                    # Windows 下 UDP 目标端口无监听时，内核回 ICMP 端口不可达，
                    # 下一次 recvfrom 会抛 ConnectionResetError(10054)。
                    # 这代表接收端未启动/已停止，无需重试，直接给出友好提示。
                    self.on_message(
                        "error",
                        f"目标端口 {target_port} 无监听（接收端未启动或已停止）。"
                        "请先在「接收文件」页签点「▶️ 开始监听」，再发送文件。",
                    )
                    sock.close()
                    return False
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
                except ConnectionResetError:
                    self.on_message("error", "接收端连接已断开（可能已停止监听），传输中止")
                    sock.close()
                    return False
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
            total_retransmissions = sum(self._retry_count.values())
            self.on_message(
                "success",
                f"文件发送完成! 总时间: {total_time:.2f}秒, 平均速度: {total_speed:.2f} MB/s, 重传 {total_retransmissions} 次",
            )
            sock.close()

            return self.save_transmission_data(
                file_data, filename, "sent", modulation_type, coding_scheme, snr_db,
                duration=total_time, speed=total_speed, retransmissions=total_retransmissions,
                total_chunks=total_chunks,
            )

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
        duration: Optional[float] = None,
        speed: Optional[float] = None,
        retransmissions: Optional[int] = None,
        total_chunks: Optional[int] = None,
        completion_ratio: Optional[float] = None,
        missing_chunks: Optional[int] = None,
    ) -> Dict[str, Any]:
        """保存传输数据（按 ``sample_size`` 采样），返回数据记录，并备份到临时文件。

        ``duration / speed / retransmissions`` 等为传输性能指标，供误码率分析页展示。
        """
        modulation_type = modulation_type or self.modem.modulation_type
        coding_scheme = coding_scheme or "重复编码"
        snr_db = snr_db if snr_db is not None else self.modem.snr_db

        sample_data = file_data if self.sample_size is None else file_data[:self.sample_size]
        bits = self.bytes_to_bits(sample_data)

        record = {
            "original_bits": bits,
            "filename": filename,
            "file_size": len(file_data),
            "analyzed_bytes": len(sample_data),
            "timestamp": time.time(),
            "modulation_type": modulation_type,
            "coding_scheme": coding_scheme,
            "snr_db": snr_db,
            "duration": duration,
            "speed": speed,
            "retransmissions": retransmissions,
            "total_chunks": total_chunks,
            "completion_ratio": completion_ratio,
            "missing_chunks": missing_chunks,
        }

        try:
            os.makedirs("./temp_analysis", exist_ok=True)
            temp_file = f"./temp_analysis/{direction}_{filename}_{int(time.time())}.dat"
            with open(temp_file, "wb") as f:
                f.write(sample_data)
        except Exception as e:
            self._notify("error", f"保存传输数据错误: {e}")

        self._notify("success", f"✅ {direction}数据已保存用于误码率分析 (已备份到文件)")
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
                "analyzed_bytes": len(file_data),
                "timestamp": os.path.getmtime(latest_file),
                "modulation_type": self.modem.modulation_type,
                "coding_scheme": "重复编码",
                "snr_db": self.modem.snr_db,
                # 性能指标未持久化到 .dat 文件，从文件恢复时置 None
                "duration": None,
                "speed": None,
                "retransmissions": None,
                "total_chunks": None,
                "completion_ratio": None,
                "missing_chunks": None,
            }
        except Exception as e:
            self.on_message("error", f"从文件加载数据错误: {e}")
            return None

    # ---------- 接收 ----------
    def stop_receiver(self) -> None:
        """停止接收端：设置停止事件并关闭 socket，使阻塞的 recvfrom 立即返回。"""
        self._receiver_stop.set()
        # 立即反映"未监听"；线程 finally 中还有 generation 兜底，不会误清新线程的状态
        self._receiver_alive.clear()
        sock = self._receiver_socket
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def start_receiver(self, listen_port: int, save_dir: str):
        """启动接收端，接收文件并返回用于误码率分析的数据记录。

        持续监听直到收到 ``END_OF_TRANSMISSION`` 完成一次传输，或调用
        :meth:`stop_receiver` 主动停止。若 ``listen_port`` 被占用则自动尝试
        下一个可用端口，并通过 ``receiver_actual_port`` 暴露实际端口。
        """
        sock: Optional[socket.socket] = None
        self._receiver_stop.clear()
        self.receiver_error = None
        self.receiver_actual_port = None
        self._receiver_thread_ident = threading.get_ident()
        with self._receiver_lock:
            self._receiver_generation += 1
            generation = self._receiver_generation
            self._receiver_alive.set()
        try:
            os.makedirs(save_dir, exist_ok=True)

            # 绑定端口；失败则自动尝试下一个可用端口
            bound_port: Optional[int] = None
            max_attempts = 10
            for attempt in range(max_attempts):
                candidate = listen_port + attempt
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
                    sock.bind(("0.0.0.0", candidate))
                    sock.settimeout(1.0)
                    bound_port = candidate
                    break
                except OSError:
                    if sock is not None:
                        sock.close()
                        sock = None
                    continue

            if bound_port is None:
                self.receiver_error = f"端口 {listen_port}~{listen_port + max_attempts - 1} 均被占用，无法监听"
                self._receiver_log("error", self.receiver_error)
                return None

            assert sock is not None  # bound_port 非空说明 bind 已成功，sock 必非 None
            self._receiver_socket = sock
            self.receiver_actual_port = bound_port
            if bound_port != listen_port:
                self._receiver_log("info", f"端口 {listen_port} 被占用，已自动改用端口 {bound_port} 监听")

            self._receiver_log("success", f"开始在端口 {bound_port} 监听...")

            received_data: Dict[int, bytes] = {}
            total_chunks = 0
            filename = ""
            start_time = time.time()

            receiver_modulation_type = "BPSK"
            receiver_coding_scheme = "重复编码"
            receiver_snr_db = 10.0

            self._receiver_update_progress(0, 1)
            self._reset_state()

            while not self._receiver_stop.is_set():
                try:
                    data, addr = sock.recvfrom(self.chunk_size + 8)

                    if data.startswith(b"FILE_INFO"):
                        info_parts = data.decode().split("|")
                        filename = info_parts[1]
                        total_chunks = int(info_parts[3])

                        if len(info_parts) >= 7:
                            receiver_modulation_type = info_parts[4]
                            receiver_coding_scheme = info_parts[5]
                            receiver_snr_db = float(info_parts[6])
                            self._receiver_log(
                                "info",
                                f"📡 从发送端接收参数: 调制={receiver_modulation_type}, "
                                f"编码={receiver_coding_scheme}, 信噪比={receiver_snr_db}dB",
                            )

                        sock.sendto(b"ACK_FILE_INFO", addr)
                        self._receiver_update_status(f"开始接收文件: {filename}")
                        start_time = time.time()
                        continue

                    if data == b"END_OF_TRANSMISSION":
                        self._receiver_log("success", "传输完成确认收到")
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
                                self._receiver_update_progress(current, total_chunks)
                                elapsed = time.time() - start_time
                                received_size = sum(len(d) for d in received_data.values())
                                speed = (received_size / 1024 / 1024) / elapsed if elapsed > 0 else 0
                                self._receiver_update_status(f"接收进度: {current}/{total_chunks} | 速度: {speed:.2f} MB/s")

                            if len(received_data) >= total_chunks > 0:
                                break

                        except struct.error as e:
                            self._receiver_log("warning", f"数据包解析错误: {e}")
                            continue

                except socket.timeout:
                    # 一直监听：空闲超时不退出，继续等待下一个数据包
                    continue

                except OSError as e:
                    # recvfrom 抛 OSError 通常意味着 socket 已被 stop_receiver 关闭，直接退出
                    if not self._receiver_stop.is_set():
                        self._receiver_log("error", f"接收错误: {str(e)}")
                    break

                except Exception as e:
                    self._receiver_log("error", f"接收错误: {str(e)}")
                    if len(received_data) >= total_chunks > 0:
                        break
                    continue

            # 重组文件（即使不完整也保存）
            if not received_data:
                if self._receiver_stop.is_set():
                    self._receiver_log("info", "接收端已停止")
                else:
                    self._receiver_log("error", "未接收到有效数据")
                return None

            sorted_data: List[bytes] = []
            missing_chunks: List[int] = []
            for i in range(total_chunks):
                if i in received_data:
                    sorted_data.append(received_data[i])
                else:
                    missing_chunks.append(i)

            if missing_chunks:
                self._receiver_log("warning", f"缺失 {len(missing_chunks)} 个数据块: {missing_chunks}")

            file_data = b"".join(sorted_data)

            save_path = os.path.join(save_dir, f"received_{filename}")
            try:
                with open(save_path, "wb") as f:
                    f.write(file_data)
                self._receiver_log("success", f"文件保存成功: {save_path}")
            except Exception as save_error:
                self._receiver_log("error", f"文件保存失败: {save_error}")

            elapsed = time.time() - start_time
            file_size_mb = len(file_data) / 1024 / 1024
            speed = file_size_mb / elapsed if elapsed > 0 else 0
            completion_ratio = len(received_data) / total_chunks if total_chunks > 0 else 0

            if completion_ratio < 1.0:
                self._receiver_log("warning", f"文件接收不完整: {completion_ratio:.1%}")
            self._receiver_log(
                "info",
                f"传输统计: 大小 {file_size_mb:.2f} MB, 时间 {elapsed:.2f} 秒, "
                f"平均速度 {speed:.2f} MB/s, 完整度 {completion_ratio:.1%}",
            )

            return self.save_transmission_data(
                file_data, filename, "received",
                receiver_modulation_type, receiver_coding_scheme, receiver_snr_db,
                duration=elapsed, speed=speed, total_chunks=total_chunks,
                completion_ratio=completion_ratio, missing_chunks=len(missing_chunks),
            )

        except Exception as e:
            self.receiver_error = f"启动接收端失败: {str(e)}"
            self._receiver_log("error", self.receiver_error)
            return None
        finally:
            # generation 兜底：仅当没有更新的接收线程启动时才清除 alive，避免误清新线程状态
            with self._receiver_lock:
                if self._receiver_generation == generation:
                    self._receiver_alive.clear()
            self._receiver_socket = None
            self._receiver_thread_ident = None
            self._receiver_stop.clear()
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
