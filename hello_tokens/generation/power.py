"""Keep Windows from throttling this process while it measures or writes.

Windows applies "power throttling" (EcoQoS) to processes it judges to be background work: after
about a second of steady running, it lowers their CPU speed to save power. Generation here is limited
by how fast the CPU launches GPU work, so throttling made each token about 3x slower (6 ms -> 18 ms),
and a benchmark would measure the power policy instead of the code.

A process may ask not to be throttled. This changes nothing system-wide, only this process, and
only until it exits.
"""

import ctypes
import sys

_PROCESS_POWER_THROTTLING = 4  # PROCESS_INFORMATION_CLASS: ProcessPowerThrottling
_EXECUTION_SPEED = 1  # PROCESS_POWER_THROTTLING_EXECUTION_SPEED


class _ThrottlingState(ctypes.Structure):
    _fields_ = [("version", ctypes.c_ulong), ("control_mask", ctypes.c_ulong), ("state_mask", ctypes.c_ulong)]


def opt_out_of_power_throttling() -> bool:
    """Ask Windows not to slow this process down. Returns True if the request was accepted."""
    if sys.platform != "win32":
        return False  # other systems don't have this policy
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    # Control the execution-speed setting (control_mask) and set it to off (state_mask 0).
    state = _ThrottlingState(1, _EXECUTION_SPEED, 0)
    return bool(kernel32.SetProcessInformation(
        kernel32.GetCurrentProcess(), _PROCESS_POWER_THROTTLING, ctypes.byref(state), ctypes.sizeof(state)))
