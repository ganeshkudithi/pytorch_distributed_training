"""
monitoring/gpu.py
=================

GPU and System monitoring utilities.

Responsibilities
----------------
1. GPU utilization
2. GPU memory usage
3. CPU RAM usage
4. GPU name
5. GPU temperature
"""

from __future__ import annotations

import psutil

from pynvml import (
    nvmlInit,
    nvmlShutdown,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetTemperature,
    nvmlDeviceGetName,
    NVML_TEMPERATURE_GPU,
)


_initialized = False


def initialize_nvml():
    """
    Initialize NVML.
    """

    global _initialized

    if not _initialized:
        nvmlInit()
        _initialized = True


def shutdown_nvml():
    """
    Shutdown NVML.
    """

    global _initialized

    if _initialized:
        nvmlShutdown()
        _initialized = False


def _get_handle(device_index: int = 0):
    """
    Return NVML device handle.
    """

    initialize_nvml()

    return nvmlDeviceGetHandleByIndex(device_index)


def get_gpu_name(device_index: int = 0) -> str:
    """
    Get GPU name.
    """

    handle = _get_handle(device_index)

    return nvmlDeviceGetName(handle)


def get_gpu_utilization(device_index: int = 0) -> float:
    """
    GPU utilization (%)
    """

    handle = _get_handle(device_index)

    utilization = nvmlDeviceGetUtilizationRates(handle)

    return float(utilization.gpu)


def get_gpu_memory(device_index: int = 0) -> float:
    """
    GPU memory usage (GB)
    """

    handle = _get_handle(device_index)

    memory = nvmlDeviceGetMemoryInfo(handle)

    return memory.used / (1024 ** 3)


def get_gpu_memory_total(device_index: int = 0) -> float:
    """
    Total GPU memory (GB)
    """

    handle = _get_handle(device_index)

    memory = nvmlDeviceGetMemoryInfo(handle)

    return memory.total / (1024 ** 3)


def get_gpu_memory_free(device_index: int = 0) -> float:
    """
    Free GPU memory (GB)
    """

    handle = _get_handle(device_index)

    memory = nvmlDeviceGetMemoryInfo(handle)

    return memory.free / (1024 ** 3)


def get_gpu_temperature(device_index: int = 0) -> int:
    """
    GPU temperature (°C)
    """

    handle = _get_handle(device_index)

    return nvmlDeviceGetTemperature(
        handle,
        NVML_TEMPERATURE_GPU,
    )


def get_cpu_memory() -> float:
    """
    System RAM usage (GB)
    """

    memory = psutil.virtual_memory()

    used = memory.used / (1024 ** 3)

    return used


def get_cpu_memory_percent() -> float:
    """
    System RAM usage (%)
    """

    memory = psutil.virtual_memory()

    return float(memory.percent)


def get_gpu_stats(device_index: int = 0) -> dict:
    """
    Return all GPU statistics.
    """

    return {
        "gpu_name": get_gpu_name(device_index),
        "gpu_utilization": get_gpu_utilization(device_index),
        "gpu_memory_used_gb": get_gpu_memory(device_index),
        "gpu_memory_total_gb": get_gpu_memory_total(device_index),
        "gpu_memory_free_gb": get_gpu_memory_free(device_index),
        "gpu_temperature": get_gpu_temperature(device_index),
        "cpu_memory_used_gb": get_cpu_memory(),
        "cpu_memory_percent": get_cpu_memory_percent(),
    }
