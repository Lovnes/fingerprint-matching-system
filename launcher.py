"""Start FP-Match and open it in the default browser.

Used by the double-click launchers (FP-Match.command on macOS, FP-Match.bat on Windows)
and as the entry point of the packaged app built with PyInstaller (see packaging/).

    python launcher.py            # same as double-clicking
    FPMATCH_PORT=8600 python launcher.py
"""
from __future__ import annotations

import multiprocessing
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser

PREFERRED_PORT = int(os.environ.get("FPMATCH_PORT", "8501"))
FROZEN = getattr(sys, "frozen", False)


def base_dir() -> str:
    """Folder holding app.py: the bundle's data folder when packaged, else this file's folder."""
    return getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))


def healthy(port: int, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/_stcore/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def open_when_ready(port: int):
    url = f"http://localhost:{port}"
    for _ in range(240):  # first start of a packaged app can be slow (antivirus scans, cold disk)
        if healthy(port, 0.5):
            webbrowser.open(url)
            return
        time.sleep(0.5)
    print(f"FP-Match did not respond. Try opening {url} yourself.", file=sys.stderr)


def redirect_output_to_log():
    """A windowed app has no console, so keep a log for troubleshooting."""
    from fpmatch.database import user_data_dir

    os.makedirs(user_data_dir(), exist_ok=True)
    log = open(os.path.join(user_data_dir(), "fpmatch.log"), "a", buffering=1, encoding="utf-8")
    sys.stdout = sys.stderr = log


def main():
    if FROZEN:
        # Load OpenCV now, while sys.path is still PyInstaller's. Streamlit later puts the bundle folder
        # (which also holds the cv2 package) first on sys.path, and OpenCV's loader then finds its own
        # package instead of the native module and fails with "recursion is detected".
        import cv2  # noqa: F401
        if sys.stdout is None or sys.platform == "darwin":
            redirect_output_to_log()

    # Second launch while already running: just bring the browser back.
    if healthy(PREFERRED_PORT):
        webbrowser.open(f"http://localhost:{PREFERRED_PORT}")
        return

    port = PREFERRED_PORT
    while not port_free(port) and port < PREFERRED_PORT + 50:
        port += 1

    os.environ["FPMATCH_LAUNCHER"] = "1"  # shows the Quit button in the sidebar
    print("=" * 64)
    print(f" FP-Match is running at http://localhost:{port}")
    print(" Keep this window open while you use it. Close it (or use")
    print(" 'Quit FP-Match' in the sidebar) to stop.")
    print("=" * 64, flush=True)
    threading.Thread(target=open_when_ready, args=(port,), daemon=True).start()

    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit", "run", os.path.join(base_dir(), "app.py"),
        "--global.developmentMode=false",
        "--server.address=localhost",        # never expose prints on the network
        f"--server.port={port}",
        "--server.headless=true",            # we open the browser ourselves; no e-mail prompt
        "--server.fileWatcherType=none",
        "--browser.gatherUsageStats=false",  # no telemetry: everything stays on this computer
        "--client.toolbarMode=minimal",
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
