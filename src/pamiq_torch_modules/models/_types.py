from typing import TypeAlias

from torch import Size

SizeType: TypeAlias = Size | list[int] | tuple[int, ...]
