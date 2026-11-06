"""Inspect an HA parquet before writing a head-IMU adapter.

No conversion is attempted until actual columns, timestamp origin, units, head
sensor identity, and axis definitions have been reviewed. This prevents silently
turning wrist IMU or poses into purported head gyroscope measurements.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq


def inspect_columns(path: Path) -> list[str]:
    schema = pq.read_schema(path)
    print("Parquet columns and types:")
    print(schema)
    print("File metadata:", schema.metadata)
    for field in schema:
        if field.metadata:
            print("Field metadata:", field.name, field.metadata)
    print("STOP: confirm head IMU fields, time origin, units and axes before conversion.")
    return schema.names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path)
    args = parser.parse_args(argv)
    inspect_columns(args.parquet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
