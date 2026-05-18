from __future__ import annotations

import argparse
from pathlib import Path


def build_tree(path: Path, prefix: str = "") -> None:
    entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    last_index = len(entries) - 1

    for index, entry in enumerate(entries):
        is_last = index == last_index
        branch = "└── " if is_last else "├── "
        print(f"{prefix}{branch}{entry.name}")

        if entry.is_dir():
            extension = "    " if is_last else "│   "
            build_tree(entry, prefix + extension)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print a directory tree for the given folder.")
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Folder path to print the tree for (default: current folder)",
    )
    args = parser.parse_args()

    root = Path(args.path).resolve()
    if not root.exists():
        raise SystemExit(f"Error: path does not exist: {root}")
    if root.is_file():
        raise SystemExit(f"Error: path must be a directory: {root}")

    print(root.name)
    build_tree(root)
