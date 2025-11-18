from __future__ import annotations
from threading import Thread
from typing import Any


class TimeoutError(Exception):
    """Raised when the sandboxed execution exceeds the time-limit."""
    pass


def run_with_timeout(func, args: tuple[Any, ...] = (), timeout: int = 15):
    """
    Execute `func(*args)` in a daemon thread.
    Raise `TimeoutError` if it runs longer than `timeout` seconds.
    """
    result: list[Any] = []
    exception: list[BaseException] = []

    def target():
        try:
            result.append(func(*args))
        except BaseException as e:
            exception.append(e)

    thread = Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise TimeoutError(f"Execution timed out after {timeout}s")

    if exception:
        raise exception[0]

    return result[0] if result else None 