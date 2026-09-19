"""Validated target policy, shared by launch, verifier and compilation cache."""
from dataclasses import dataclass

from .errors import TilaError, TilaLaunchContractError


@dataclass(frozen=True)
class Capability:
    name: str = "NVIDIA GeForce RTX 3090"
    arch: tuple = (8, 6)
    triton_version: str = "3.6.0"
    grid_limits: tuple = ((1 << 31) - 1, 65535, 65535)
    num_warps: tuple = (4, 8)
    max_tile_elements: int = 1 << 20
    dot_inputs: tuple = ("f16",)
    dot_outputs: tuple = ("f16", "f32")
    min_dot_k: int = 16
    fp8_storage: bool = False


SUPPORTED = Capability()


@dataclass(frozen=True)
class CUDATarget:
    device_index: int
    uuid: str
    torch_version: str
    cuda_version: str
    capability: Capability = SUPPORTED


def validate_options(num_warps):
    if type(num_warps) is not int or num_warps not in SUPPORTED.num_warps:
        raise TilaError("TILA-TARGET-006", "num_warps must be exact int 4 or 8 on the validated target")


def validate_grid(grid):
    """CPU shares the GPU launch domain; zero axes mean no programs."""
    if not isinstance(grid, tuple) or not 1 <= len(grid) <= 3:
        raise TilaLaunchContractError("TILA-TYPE-104", "grid must have one to three axes")
    for axis, (value, limit) in enumerate(zip(grid, SUPPORTED.grid_limits)):
        if type(value) is not int or not 0 <= value <= limit:
            raise TilaLaunchContractError("TILA-TYPE-104",
                f"grid[{axis}] must be an exact int in [0, {limit}]")


def resolve_cuda(torch, triton, device):
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != SUPPORTED.arch or props.name != SUPPORTED.name:
        raise TilaError("TILA-TARGET-007", "unvalidated CUDA target; current support is RTX 3090 / SM86")
    if triton.__version__ != SUPPORTED.triton_version:
        raise TilaError("TILA-TARGET-007", "unvalidated Triton version; current support requires 3.6.0")
    return CUDATarget(device.index, str(props.uuid), torch.__version__, torch.version.cuda)
