"""Compare old/new MERN parity JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _flatten(prefix, value, out):
    if isinstance(value, dict):
        for key, child in value.items():
            _flatten(f"{prefix}.{key}" if prefix else key, child, out)
    elif isinstance(value, list):
        out[prefix] = value
    elif isinstance(value, (int, float)):
        out[prefix] = float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--label", default="repro parity")
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()

    old = json.loads(args.old.read_text())
    new = json.loads(args.new.read_text())
    old_flat = {}
    new_flat = {}
    _flatten("", old, old_flat)
    _flatten("", new, new_flat)

    keys = sorted(set(old_flat) & set(new_flat))
    failures = []
    for key in keys:
        old_value = old_flat[key]
        new_value = new_flat[key]
        if isinstance(old_value, list) or isinstance(new_value, list):
            if old_value != new_value:
                failures.append((key, old_value, new_value, None))
            continue
        if not np.isclose(old_value, new_value, rtol=args.rtol, atol=args.atol):
            failures.append((key, old_value, new_value, abs(old_value - new_value)))

    print(f"Checking {args.label}")
    print(f"Compared {len(keys)} shared numeric/list fields")
    if failures:
        print(f"{args.label} failed: found {len(failures)} mismatches")
        for key, old_value, new_value, diff in failures[:50]:
            if diff is None:
                print(f"- {key}: old={old_value!r} new={new_value!r}")
            else:
                print(f"- {key}: old={old_value:.12g} new={new_value:.12g} diff={diff:.3g}")
        raise SystemExit(1)
    print(f"{args.label} passed")


if __name__ == "__main__":
    main()
