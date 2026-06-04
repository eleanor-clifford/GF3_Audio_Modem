# vim: set ts=4 sw=4 tw=0 et :
"""
Dataclasses and parser for the generalised audio modem standard config (config.yaml).

Each standard is defined as a top-level sample_rate and a payload list of blocks.
Blocks can nest (e.g. repetition wraps other blocks).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union

import yaml


@dataclass
class ChirpBlock:
    """A linear chirp signal."""
    duration: int = 0          # samples
    frequencies: List[float] = field(default_factory=lambda: [0.0, 0.0])


@dataclass
class SilenceBlock:
    """A stretch of silence."""
    length: int = 0            # samples


@dataclass
class GolayABlock:
    """'A' part of Golay pair."""
    order: int
    seed: list[int]


@dataclass
class GolayBBlock:
    """'B' part of Golay pair."""
    order: int
    seed: list[int]


@dataclass
class KnownSymbolBlock:
    """A known OFDM symbol for synchronisation / channel estimation."""
    length: int = 0            # samples (OFDM body length)
    prefix: int = 0            # cyclic prefix length
    bins: List[int] = field(default_factory=list)  # [start, end) data bin range
    data: List[int] = field(default_factory=list)


@dataclass
class DataSymbolBlock:
    """A data-carrying OFDM symbol."""
    length: int = 0            # samples (OFDM body length)
    prefix: int = 0            # cyclic prefix length
    bins: List[int] = field(default_factory=list)  # [start, end) data bin range


# Union alias for readability
Block = Union[
    ChirpBlock,
    SilenceBlock,
    GolayABlock,
    GolayBBlock,
    KnownSymbolBlock,
    DataSymbolBlock,
    "RepetitionBlock",
]


@dataclass
class RepetitionBlock:
    """Repeat a sub-payload N times (or None = fill remaining frame)."""
    type: str = "repetition"
    number: Optional[int] = None
    payload: List[Block] = field(default_factory=list)


# --- Parser -----------------------------------------------------------------

_TYPE_MAP: dict[str, type] = {
    "chirp": ChirpBlock,
    "silence": SilenceBlock,
    "golay_a": GolayABlock,
    "golay_b": GolayBBlock,
    "known_symbol": KnownSymbolBlock,
    "data_symbol": DataSymbolBlock,
    "repetition": RepetitionBlock,
}


def _parse_block(raw: dict) -> Block:
    """Recursively convert a raw YAML dict into the appropriate dataclass."""
    block_type = raw.get("type", "")
    cls = _TYPE_MAP.get(block_type)
    if cls is None:
        raise ValueError(f"Unknown block type: {block_type!r}")

    if block_type == "repetition":
        sub_payload = raw.get("payload", [])
        return RepetitionBlock(
            type="repetition",
            number=raw.get("number"),
            payload=[_parse_block(b) for b in sub_payload],
        )

    # For all other types, pass matching kwargs to the constructor.
    kwargs = {k: v for k, v in raw.items() if k != "type"}
    return cls(**kwargs)


def _flatten(payload: list[Block]) -> list[Block]:
    """Flatten repetition blocks with a known count into a flat list.

    Repetition blocks with number=None are kept as-is (unknown count).
    """
    flat = []
    for block in payload:
        if isinstance(block, RepetitionBlock):
            if block.number is not None:
                for _ in range(block.number):
                    flat.extend(_flatten(block.payload))
            else:
                # Unknown count -- keep the repetition block intact
                flat.append(block)
        else:
            flat.append(block)
    return flat


@dataclass
class Standard:
    """A complete modem standard definition loaded from config.yaml."""
    sample_rate: int
    payload: list[Block] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "Standard":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        raw_payload = [_parse_block(b) for b in raw["payload"]]
        return cls(
            sample_rate=raw["sample_rate"],
            payload=_flatten(raw_payload),
        )
