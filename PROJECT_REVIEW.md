# 项目复盘：接收端状态同步 Bug 排查与修复

> 记录一次真实的多线程 / 状态同步缺陷，从现象定位到根因、再到工程化修复的完整过程。
> 涉及：Python 多线程、Streamlit 后台线程限制、线程安全设计、UDP socket 异常处理。

## 一、问题现象

在「可靠 UDP 文件传输与调制解调系统」中：

1. 在「接收文件」页签点击「开始监听」，界面显示 `🔴 接收端正在监听中...`
2. 切换到「发送文件」页签点击「开始发送」，却弹出红框：`目标为本地主机，但接收端未在监听`
3. 即：**接收页看到的监听状态为「在监听」，发送页读到的却是「未监听」，两个页签状态不一致**

## 二、根因分析

通过端口状态检查（`netstat`）与 Streamlit 运行日志定位到两个叠加的缺陷：

### 1. `receiver_alive` 被两个线程竞态写入（核心根因）

`receiver_alive` 原本是 `ReliableUDPTransfer` 的一个普通实例布尔属性，存在两处写者：

| 写者 | 线程 | 写入时机 | 写入值 |
| --- | --- | --- | --- |
| UI 主线程 | main | 「开始监听」按钮处理器 | `True`（乐观置位，为了让界面立即显示） |
| 接收线程 | background | `start_receiver` 入口 / `finally` | `True` / `False` |

**竞态时序**：接收线程启动后若立即失败退出（如端口被占用导致 `bind` 失败），其 `finally` 会把 `receiver_alive` 置回 `False`；但 main 线程的「乐观置 True」可能已经渲染成「正在监听」。于是出现「接收页看是 True、发送页读是 False」的假象——实际上接收线程早已退出。

### 2. bind 失败异常被吞掉

`start_receiver` 外层 `except Exception` 里通过 `on_message("error", ...)` 报错。而 `on_message` 回调在接收线程（后台线程）里执行的是 `st.error(...)`：

```
Thread 'Thread-5 (receiver_thread)': missing ScriptRunContext! ...
```

Streamlit 后台线程没有 `ScriptRunContext`，`st.*` 调用是**静默 no-op**，不抛异常、也不显示到界面。导致：**bind 失败时错误被静默吞掉，线程静默退出，用户完全不知道原因**。

## 三、解决方案

### 1. 线程安全的状态设计（单写者 + 代际兜底）

- `receiver_alive` 由普通布尔改为 **`threading.Event`**，只读属性对外暴露（`@property`）
- **唯一写者是接收线程**：入口 `set()`、`finally` `clear()`
- 引入**代际计数器 `_receiver_generation`**：`finally` 中仅在「当前代际仍是自己」时才清除，避免旧线程退出时误清新线程的状态（解决快速 stop→start 的竞态）
- 删除 UI 层的「乐观置位」，状态不再有第二写者

### 2. 错误暴露到界面

- 新增 `receiver_error` 字段，bind 失败 / 启动异常时由接收线程写入
- UI 主线程在每次渲染时读取并 `st.error(...)` 展示，绕开「后台线程调 st.* 失效」的问题

### 3. 接收线程不再直接调用 st.*

- 接收线程的 `on_message/on_progress/on_status` 全部改为写入**线程安全的缓冲**（`threading.Lock` 保护的消息/进度/状态队列）
- UI 通过 `st.fragment(run_every="1s")` 每秒自动刷新，`drain_receiver_events()` 取出缓冲并渲染——消息/进度/状态第一次真正显示到了界面上
- 接收线程产生的数据记录通过 `set/take_last_received_record()` 线程安全地交给主线程写入 `session_state`（后台线程不直接写 session_state）

### 4. 端口占用自动切换

- `start_receiver` 绑定失败时自动尝试下一个可用端口（最多 10 个）
- 实际端口通过 `receiver_actual_port` 暴露到界面，并提示「端口 X 被占用，已自动改用端口 Y」

## 四、工程能力体现

| 能力点 | 具体体现 |
| --- | --- |
| **并发与线程安全** | 识别出「多线程共享可变状态」的竞态，用 `threading.Event` 单写者 + 代际计数器消除状态撕裂 |
| **框架限制的认知** | 理解 Streamlit 后台线程无 `ScriptRunContext`，`st.*` 调用失效；用「线程安全缓冲 + 主线程 drain」解耦后台线程与 UI |
| **异常处理** | 区分 Windows UDP 的 `ConnectionResetError(10054)` 与 `socket.timeout`，按语义给出友好提示而非让异常冒泡 |
| **问题定位方法** | 通过 `netstat` 检查端口、读运行日志找 `missing ScriptRunContext` 证据、最小复现脚本（向关闭端口发 UDP）实锤根因 |
| **可测试性** | 核心逻辑（自动换端口、状态生命周期、缓冲）可脱离 UI 用冒烟脚本验证 |

## 五、关键代码位置

- `src/transport.py`
  - `__init__`：接收端状态字段与线程安全缓冲
  - `receiver_alive`（property）+ `_receiver_generation`
  - `start_receiver`：Event 状态、自动换端口、缓冲输出、代际兜底
  - `stop_receiver` / `drain_receiver_events` / `set/take_last_received_record`
- `app.py`
  - `_render_receiver_panel`（`st.fragment(run_every="1s")` 实时面板）
  - 「开始监听 / 停止监听」按钮处理器
