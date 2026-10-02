"""Diagnostic-only CUPTI flush on Python and multiprocessing worker shutdown."""

import atexit
import ctypes
import multiprocessing.util
import os

_registered = False


def register() -> None:
    """Flush an already-loaded collector; never load CUDA in parent processes."""
    global _registered
    path = os.environ.get("CUDA_INJECTION64_PATH")
    if _registered or os.environ.get("EFFECT_TORCH_CUPTI_FINALIZE") != "1" or not path:
        return
    _registered = True
    pid = os.getpid()

    def flush() -> None:
        # atexit callbacks can be inherited by forked children. Each spawned
        # process registers its own callback and multiprocessing Finalize pid.
        if os.getpid() != pid:
            return
        try:
            collector = ctypes.CDLL(path, mode=os.RTLD_NOLOAD | os.RTLD_LOCAL)
        except OSError:
            return  # This process never initialized the injected collector.
        try:
            function = collector.EffectTorchCuptiFlush
            function.argtypes = []
            function.restype = None
            function()
        except Exception as error:
            # Missing final summary makes the profile invalid; keep errors
            # visible without replacing the worker's own shutdown exception.
            print(f"CUPTI final flush failed in pid {pid}: {error}", file=__import__("sys").stderr)

    multiprocessing.util.Finalize(None, flush, exitpriority=100)
    atexit.register(flush)
