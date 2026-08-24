"""Early, dependency-free diagnostics for Gradmotion training entrypoints.

This module intentionally avoids importing torch or Isaac Gym so it can run
before the strict Isaac Gym import sequence begins.
"""

from __future__ import annotations

import atexit
import faulthandler
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from typing import Iterable


_STARTED_AT = time.monotonic()
_SENSITIVE_ARGUMENT = re.compile(
    r"(?i)(api[-_]?key|access[-_]?key|secret|signature|token|password|credential)"
)


def _emit(event: str, **fields: object) -> None:
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"[gm-startup] event={event}{' ' + details if details else ''}", flush=True)


def _safe_argv(argv: Iterable[str]) -> str:
    values = list(argv)
    sanitized = []
    hide_next = False
    for value in values:
        if hide_next:
            sanitized.append("<redacted>")
            hide_next = False
            continue
        if value.startswith("--") and "=" in value:
            key, raw_value = value.split("=", 1)
            sanitized.append(f"{key}=<redacted>" if _SENSITIVE_ARGUMENT.search(key) else value)
            continue
        sanitized.append(value)
        if value.startswith("--") and _SENSITIVE_ARGUMENT.search(value):
            hide_next = True
    return repr(sanitized)


def _git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception as exc:  # diagnostics must never block training
        return f"unavailable:{type(exc).__name__}"


def _gpu_summary() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return ";".join(line.strip() for line in result.stdout.splitlines() if line.strip())
    except Exception as exc:  # diagnostics must never block training
        return f"unavailable:{type(exc).__name__}"


def _unhandled_exception(exc_type, exc_value, exc_traceback) -> None:
    _emit("unhandled_exception", type=getattr(exc_type, "__name__", str(exc_type)))
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    sys.stderr.flush()


def _thread_exception(args) -> None:
    _emit(
        "thread_exception",
        thread=getattr(args.thread, "name", "unknown"),
        type=getattr(args.exc_type, "__name__", str(args.exc_type)),
    )
    traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback)
    sys.stderr.flush()


def _signal_handler(signum, _frame) -> None:
    signal_name = getattr(signal.Signals(signum), "name", str(signum))
    _emit("signal", name=signal_name, number=signum)
    raise SystemExit(128 + signum)


def _heartbeat() -> None:
    while True:
        time.sleep(30)
        _emit("heartbeat", uptime_s=round(time.monotonic() - _STARTED_AT, 1))


def install() -> None:
    """Install early diagnostics without importing training dependencies."""
    try:
        faulthandler.enable(all_threads=True)
    except Exception as exc:
        _emit("faulthandler_unavailable", error=type(exc).__name__)

    sys.excepthook = _unhandled_exception
    if hasattr(threading, "excepthook"):
        threading.excepthook = _thread_exception

    for handled_signal in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(handled_signal, _signal_handler)
        except Exception as exc:
            _emit(
                "signal_handler_unavailable",
                signal=getattr(handled_signal, "name", str(handled_signal)),
                error=type(exc).__name__,
            )

    atexit.register(
        lambda: _emit("process_exit", uptime_s=round(time.monotonic() - _STARTED_AT, 1))
    )

    try:
        disk = shutil.disk_usage(os.getcwd())
        disk_free_gib = round(disk.free / (1024 ** 3), 2)
    except Exception:
        disk_free_gib = "unavailable"

    _emit(
        "process_start",
        pid=os.getpid(),
        ppid=os.getppid(),
        python=sys.version.split()[0],
        cwd=repr(os.getcwd()),
        argv=_safe_argv(sys.argv),
    )
    _emit(
        "environment",
        git_commit=_git_revision(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        disk_free_gib=disk_free_gib,
        gpu=repr(_gpu_summary()),
    )

    threading.Thread(target=_heartbeat, name="gm-startup-heartbeat", daemon=True).start()

