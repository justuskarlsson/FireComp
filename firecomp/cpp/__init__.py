from typing import Literal, TypedDict
from torch.utils.cpp_extension import load
import os
import torch

_folder = os.path.abspath(os.path.dirname(__file__))

files = ["main.cpp"]
sources = [os.path.join(_folder, f) for f in files]


cpp = load(
    name="cpp",
    sources=sources,
    extra_cflags=["-O3"],
    extra_include_paths=[os.path.join(_folder, "include")],
    verbose=False,
)


class FireComponent(TypedDict):
    idx: int
    size: int
    num_fire: int
    t_min: int
    t_max: int
    y_min: int
    y_max: int
    x_min: int
    x_max: int


def search(x, y, width, height, cls, t) -> tuple[list[FireComponent], torch.Tensor]:
    return cpp.search(x, y, width, height, cls, t)


def group_by_fire(xytc, sizes_per_c, min_size) -> tuple[torch.Tensor, list[int]]:
    return cpp.group_by_fire(xytc, sizes_per_c, min_size)


def group_by_time(xytc) -> tuple[torch.Tensor, list[int]]:
    return cpp.group_by_time(xytc)


def group_by_block(
    block_num_x: int,
    block_num_y: int,
    block_idx: torch.Tensor,
    local_x: torch.Tensor,
    local_y: torch.Tensor,
    data: torch.Tensor,
):
    return cpp.group_by_blocks(
        block_num_x,
        block_num_y,
        block_idx,
        local_x,
        local_y,
        data,
    )


def project(
    data: torch.Tensor,
    W: int,
    H: int,
    xy_coords: torch.Tensor,
    method: Literal["nearest", "inv_dist", "max"],
    kernel_size: int,
    no_data_val: float | int,
) -> torch.Tensor:
    return cpp.project(data, W, H, xy_coords, method, kernel_size, no_data_val)


def fill_holes(
    data: torch.Tensor,
    kernel_size: int,
    no_data_val: float | int,
    method: Literal["nearest", "inv_dist", "max"],
) -> torch.Tensor:
    return cpp.fill_holes(data, kernel_size, no_data_val, method)
