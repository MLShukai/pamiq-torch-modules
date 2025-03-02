from typing import TypeAlias

from torch import Size

SizeType: TypeAlias = Size | list[int] | tuple[int, ...]


# Type alias for 2D sizes that can be specified as a single int or a tuple of ints
size_2d = int | tuple[int, int]


def size_2d_to_int_tuple(size: size_2d) -> tuple[int, int]:
    """Convert size_2d to tuple of int.

    Args:
        size: Size specification either as int or tuple of ints.

    Returns:
        Size as tuple of ints.
    """
    if isinstance(size, int):
        return (size, size)
    return size
