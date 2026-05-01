import sys
import subprocess
import threading
import os
import time

# ================= 配置区域 =================
TARGET_SCRIPT = "01-first_mcp_server.py"

# 1. 确保日志路径绝对正确 (锁定在 log.py 同级目录)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(CURRENT_DIR, "mcp_traffic.log")


# ===========================================

def log_message(direction, message):
    """强力日志记录：写文件 + 打印到控制台"""
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    log_line = f"[{timestamp}] {direction}: {message}\n"

    try:
        # 模式 'a' (追加), encoding='utf-8'
        # buffering=1 表示行缓冲，确保每行都写入
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_line)
            f.flush()  # <--- 关键：强制刷入硬盘
            os.fsync(f.fileno())  # <--- 关键：强制操作系统存盘
    except Exception as e:
        # 如果写文件失败，至少打印出来
        sys.stderr.write(f"日志写入失败: {e}\n")


def forward_stream(source, dest, direction):
    try:
        # 记录流已开启
        log_message("SYSTEM", f"开始监听流: {direction}")

        while True:
            # 逐字节读取，防止 readline 卡死
            line = source.readline()
            if not line:
                break

            # 1. 记录日志
            try:
                decoded = line.decode('utf-8').strip()
                if decoded:
                    log_message(direction, decoded)
            except:
                log_message(direction, f"<Binary: {line}>")

            # 2. 转发数据
            dest.write(line)
            dest.flush()  # <--- 关键：立刻转发给对方

    except Exception as e:
        log_message("ERROR", f"流转发异常 ({direction}): {e}")
    finally:
        log_message("SYSTEM", f"流已关闭: {direction}")
        source.close()


def main():
    # 0. 记录启动信息
    log_message("SYSTEM", f"代理脚本启动。日志路径: {LOG_FILE}")

    python_exe = sys.executable
    script_path = os.path.join(CURRENT_DIR, TARGET_SCRIPT)

    log_message("SYSTEM", f"准备启动目标脚本: {script_path}")

    if not os.path.exists(script_path):
        log_message("FATAL", f"找不到文件: {script_path}")
        return

    try:
        # 启动真实的 MCP 服务器进程
        process = subprocess.Popen(
            [python_exe, script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0  # 无缓冲
        )
        log_message("SYSTEM", f"目标进程已启动 PID: {process.pid}")
    except Exception as e:
        log_message("FATAL", f"启动子进程失败: {e}")
        return

    # 启动线程
    t1 = threading.Thread(target=forward_stream, args=(sys.stdin.buffer, process.stdin, "Claude -> Server"))
    t2 = threading.Thread(target=forward_stream, args=(process.stdout, sys.stdout.buffer, "Server -> Claude"))

    def log_stderr():
        for line in iter(process.stderr.readline, b''):
            log_message("Server ERROR", line.decode('utf-8', errors='ignore').strip())

    t3 = threading.Thread(target=log_stderr)

    t1.daemon = True
    t2.daemon = True
    t3.daemon = True

    t1.start()
    t2.start()
    t3.start()

    process.wait()
    log_message("SYSTEM", "目标进程已退出")


if __name__ == "__main__":
    # 初始化日志文件
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== 新的会话开始于 {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    main()