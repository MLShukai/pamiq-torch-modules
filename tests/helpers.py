import pytest
import torch

CPU_DEVICE = torch.device("cpu")
CUDA_DEVICE = torch.device("cuda:0")
MPS_DEVICE = torch.device("mps:0")


def get_available_devices() -> list[torch.device]:
    devices = [CPU_DEVICE]
    if torch.cuda.is_available():
        devices.append(CUDA_DEVICE)
    if torch.backends.mps.is_available():
        devices.append(MPS_DEVICE)

    return devices


parametrize_device = pytest.mark.parametrize("device", get_available_devices())
