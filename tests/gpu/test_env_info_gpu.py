"""On a GPU machine, env_info must report the GPU fields that result files depend on."""

import pytest

from scripts.env_info import collect_env_info


@pytest.mark.gpu
def test_gpu_fields_populated():
    info = collect_env_info()
    assert info["cuda_available"] is True
    assert info["gpu_name"]
    assert info["gpu_memory_gib"] > 0
    assert info["gpu_compute_capability"]
    assert info["driver"]
