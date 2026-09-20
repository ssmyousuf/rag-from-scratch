import sys
from typing import Any


def deep_getsizeof(obj: Any, seen: set[int] | None = None) -> int:
    if seen is None:
        seen = set()

    object_id = id(obj)

    if object_id in seen:
        return 0

    seen.add(object_id)

    size = sys.getsizeof(obj)

    if isinstance(obj, dict):
        size += sum(
            deep_getsizeof(key, seen)
            + deep_getsizeof(value, seen)
            for key, value in obj.items()
        )

    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(
            deep_getsizeof(item, seen)
            for item in obj
        )

    return size


def format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"

    if size < 1024 ** 2:
        return f"{size / 1024:.2f} KB"

    if size < 1024 ** 3:
        return f"{size / (1024 ** 2):.2f} MB"

    return f"{size / (1024 ** 3):.2f} GB"