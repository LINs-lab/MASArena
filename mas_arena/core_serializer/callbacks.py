import stopit
import threading
import contextvars
from typing import Optional, Union
from contextlib import contextmanager


class Callback:
    """
    a base class for callbacks
    """

    def on_error(self, exception, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        try:
            result = self.run(*args, **kwargs)
        except Exception as e:
            self.on_error(e, *args, kwargs)
            raise e
        return result

    def run(self, *args, **kwargs):
        raise NotImplementedError(f"run is not implemented for {type(self).__name__}!")


class CallbackManager:

    def __init__(self):
        self.local_data = threading.local()
        # self.local_data.callbacks = {}

    def _ensure_callbacks(self):
        if not hasattr(self.local_data, "callbacks"):
            self.local_data.callbacks = {}

    def set_callback(self, callback_type: str, callback: Callback):
        self._ensure_callbacks()
        self.local_data.callbacks[callback_type] = callback

    def get_callback(self, callback_type: str):
        self._ensure_callbacks()
        return self.local_data.callbacks.get(callback_type, None)

    def has_callback(self, callback_type: str):
        self._ensure_callbacks()
        return callback_type in self.local_data.callbacks

    def clear_callback(self, callback_type: str):
        self._ensure_callbacks()
        if callback_type in self.local_data.callbacks:
            del self.local_data.callbacks[callback_type]

    def clear_all(self):
        self._ensure_callbacks()
        self.local_data.callbacks.clear()


callback_manager = CallbackManager()

suppress_cost_logs = contextvars.ContextVar("suppress_cost_logs", default=False)
