#!/usr/bin/env python3
"""Compare GGUF files by tensor directory and metadata, without reading tensor data.

The Dory acceptance work keeps needing one question answered: is a converted file
the same MODEL as the file NVIDIA shipped, apart from dtype? This answers it from
the GGUF headers only, so it runs in seconds on an 8 GB file.

Usage:

    dory_compare_gguf.py [--limit N] BASE.gguf OTHER.gguf [OTHER.gguf ...]

Per file: tensor count, tensor bytes, and every ``dory.*`` / ``general.*`` key.
Then each OTHER is compared against BASE: tensors missing on either side, tensors
whose shape or dtype differs, keys present on only one side, and keys whose values
differ. A file whose header promises more bytes than are on disk (an interrupted
conversion, typically) is reported as UNREADABLE rather than aborting the run.

The exit status is 0 when every comparison is clean apart from dtype, else 1.
"Clean" is a structural report, not a verdict on numerical correctness: a
converter that writes the right names with wrong data passes here.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from gguf import GGUFReader, ReaderField, ReaderTensor
from gguf.constants import GGUFValueType

REPORTED_PREFIXES = ("dory.", "general.")
STRICT_PREFIXES = ("dory.", "general.architecture")

Tensors = dict[str, tuple[tuple[int, ...], str, int]]
Keys = dict[str, object]
Loaded = tuple[Path, Tensors, Keys, str]


@dataclass(frozen=True, eq=False)
class ArrayValue:
    """An array-typed header value, kept as raw bytes so two files compare exactly."""

    count: int
    blob: bytes

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ArrayValue) and self.blob == other.blob

    def __repr__(self) -> str:
        return f"<array of {self.count} elements>"


def tensor_dims(tensor: ReaderTensor) -> tuple[int, ...]:
    """GGUF lists dimensions slowest-first; return them in conventional order."""
    return tuple(int(dim) for dim in reversed(tensor.shape))


def field_value(field: ReaderField) -> object:
    """Decode one header field's VALUE.

    In this gguf version ``parts`` starts with the key's own length, bytes and
    type tag, followed by the value parts; the key header is dropped here.
    Scalars and strings come back as plain values, arrays as an ArrayValue.
    """
    parts = list(field.parts)
    if len(parts) >= 3 and parts[1].dtype == np.uint8 and int(parts[0][0]) == len(parts[1]):
        parts = parts[3:]
    types = list(field.types)
    if types == [GGUFValueType.STRING]:
        return bytes(parts[1]).decode("utf-8")
    if len(types) == 1 and len(parts) == 1:
        flat = parts[0].reshape(-1)
        return flat[0].item() if flat.size == 1 else flat.tolist()
    count = int(parts[0][0]) if parts else 0
    blob = b"".join(np.ascontiguousarray(part).view(np.uint8).tobytes() for part in parts)
    return ArrayValue(count, blob)


def short(value: object) -> object:
    if isinstance(value, list) and len(value) > 8:
        return f"<list of {len(value)}>"
    return value


def load(path: Path) -> tuple[Tensors, Keys, str]:
    """Read one GGUF's tensor directory and every header key. Tensor data stays untouched.

    Returns an empty directory and a non-empty error string when the file cannot be
    read (for example a truncated conversion whose tensor offsets run past EOF).
    """
    try:
        reader = GGUFReader(path)
    except ValueError as exc:
        return {}, {}, f"{type(exc).__name__}: {exc} (file on disk: {path.stat().st_size:,} bytes)"
    tensors = {
        tensor.name: (tensor_dims(tensor), tensor.tensor_type.name, int(tensor.data.nbytes))
        for tensor in reader.tensors
    }
    keys = {key: field_value(field) for key, field in reader.fields.items()}
    return tensors, keys, ""


def describe(path: Path, tensors: Tensors, keys: Keys, error: str) -> None:
    print(f"== {path}")
    if error:
        print(f"   UNREADABLE: {error}")
        print("   (compare its size on disk with the other files; a short file with a full header is a truncated write)")
        print()
        return
    reported = sorted(key for key in keys if key.startswith(REPORTED_PREFIXES))
    total_bytes = sum(entry[2] for entry in tensors.values())
    print(f"   tensors: {len(tensors)}   tensor bytes: {total_bytes:,}   dory.*/general.* keys: {len(reported)}")
    for key in reported:
        print(f"     {key} = {short(keys[key])}")
    print()


def describe_difference(left: object, right: object) -> str:
    if isinstance(left, ArrayValue) or isinstance(right, ArrayValue):
        return "array contents differ"
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return f"lists differ in length ({len(left)} vs {len(right)})"
        differing = sum(1 for a, b in zip(left, right, strict=False) if a != b)
        return f"lists differ at {differing} of {len(left)} elements"
    return f"{short(left)!r} vs {short(right)!r}"


def compare(base: Loaded, other: Loaded, limit: int) -> bool:
    """Print one OTHER-vs-BASE comparison. Returns True when it is clean apart from dtype."""
    base_path, base_tensors, base_keys, base_error = base
    other_path, other_tensors, other_keys, other_error = other
    print(f"== {other_path.name}  vs  {base_path.name}")
    if base_error or other_error:
        print(f"   VERDICT: DIFFERS (unreadable: {other_error or base_error})")
        print()
        return False
    shared = sorted(base_tensors.keys() & other_tensors.keys())
    missing = sorted(base_tensors.keys() - other_tensors.keys())
    extra = sorted(other_tensors.keys() - base_tensors.keys())
    reshaped = [name for name in shared if base_tensors[name][0] != other_tensors[name][0]]
    transitions = Counter(
        f"{base_tensors[name][1]} -> {other_tensors[name][1]}"
        for name in shared
        if base_tensors[name][1] != other_tensors[name][1]
    )
    print(
        f"   tensors: {len(base_tensors)} vs {len(other_tensors)} | missing: {len(missing)}"
        f" | extra: {len(extra)} | shape mismatches: {len(reshaped)}"
    )
    for label, names in (("missing", missing), ("extra", extra), ("shape mismatch", reshaped)):
        source = other_tensors if label == "extra" else base_tensors
        for name in names[:limit]:
            print(f"     {label}: {name} {source[name][0]} {source[name][1]}")
        if len(names) > limit:
            print(f"     {label}: ... {len(names) - limit} more")
    for transition, count in sorted(transitions.items()):
        print(f"   dtype: {count} tensors {transition}")
    if not transitions:
        print("   dtype: identical")

    only_base = sorted(base_keys.keys() - other_keys.keys())
    only_other = sorted(other_keys.keys() - base_keys.keys())
    differing = sorted(key for key in base_keys.keys() & other_keys.keys() if base_keys[key] != other_keys[key])
    for label, names in (("key only in base", only_base), ("key only in other", only_other)):
        source = base_keys if label.endswith("base") else other_keys
        for key in [name for name in names if name.startswith(REPORTED_PREFIXES)][:limit]:
            print(f"     {label}: {key} = {short(source[key])}")
        unreported = sum(1 for name in names if not name.startswith(REPORTED_PREFIXES))
        if unreported:
            print(f"     {label}: ... plus {unreported} keys outside dory.*/general.*")
    for key in [name for name in differing if name.startswith(REPORTED_PREFIXES)][:limit]:
        print(f"   value differs: {key}: {describe_difference(base_keys[key], other_keys[key])}")
    unreported_differing = [name for name in differing if not name.startswith(REPORTED_PREFIXES)]
    if unreported_differing:
        shown = ", ".join(unreported_differing[:3])
        print(f"   value differs outside dory.*/general.*: {len(unreported_differing)} keys, e.g. {shown}")

    strict = [key for key in only_base + only_other + differing if key.startswith(STRICT_PREFIXES)]
    clean = not (missing or extra or reshaped or strict)
    print(f"   VERDICT: {'CLEAN apart from dtype' if clean else 'DIFFERS'}")
    print()
    return clean


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare GGUF tensor directories and header metadata.")
    parser.add_argument("--limit", type=int, default=20, help="maximum examples to list per difference class")
    parser.add_argument("base", type=Path, help="file every other file is compared against")
    parser.add_argument("others", type=Path, nargs="+", help="files to compare against base")
    args = parser.parse_args(argv)

    loaded = [(path, *load(path)) for path in [args.base, *args.others]]
    for path, tensors, keys, error in loaded:
        describe(path, tensors, keys, error)
    # A list, not a generator inside all(): every comparison must run and print.
    verdicts = [compare(loaded[0], other, args.limit) for other in loaded[1:]]
    clean = all(verdicts)
    print(f"OVERALL: {'CLEAN apart from dtype' if clean else 'DIFFERS'} ({sum(verdicts)} of {len(verdicts)} clean)")
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
