"""Top-level MERN command-line parser."""

from __future__ import annotations

import argparse

from ._train import add_train_parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mern", description="MERN command-line tools")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_train_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
