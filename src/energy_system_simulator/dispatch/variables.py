from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class VariableBlock:
    """A contiguous indexed variable block."""

    name: str
    size: int
    offset: int
    asset_id: str | None = None
    binary: bool = False


class VariableRegistry:
    """Deterministic variable index registry with optional asset dimensions."""

    def __init__(self) -> None:
        self._blocks: dict[tuple[str, str | None], VariableBlock] = {}
        self._order: list[tuple[str, str | None]] = []
        self.size = 0

    def add(
        self,
        name: str,
        size: int,
        *,
        asset_id: str | None = None,
        binary: bool = False,
    ) -> None:
        key = (name, asset_id)
        if key in self._blocks:
            raise ValueError(f"Duplicate variable block: {name}, {asset_id}")
        self._blocks[key] = VariableBlock(
            name=name,
            size=size,
            offset=self.size,
            asset_id=asset_id,
            binary=binary,
        )
        self._order.append(key)
        self.size += size

    def at(self, name: str, period: int, *, asset_id: str | None = None) -> int:
        block = self._blocks[(name, asset_id)]
        if period < 0 or period >= block.size:
            raise IndexError(f"Period {period} is outside variable block {name}")
        return block.offset + period

    def values(
        self,
        solution: FloatArray,
        name: str,
        *,
        asset_id: str | None = None,
    ) -> FloatArray:
        block = self._blocks[(name, asset_id)]
        return solution[block.offset : block.offset + block.size]

    def integrality(self) -> npt.NDArray[np.int_]:
        values = np.zeros(self.size, dtype=int)
        for block in self._blocks.values():
            if block.binary:
                values[block.offset : block.offset + block.size] = 1
        return values

    def variable_counts_by_block(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for key in self._order:
            block = self._blocks[key]
            label = block.name if block.asset_id is None else f"{block.name}[{block.asset_id}]"
            counts[label] = block.size
        return counts
