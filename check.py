#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_LINE_LIMIT = 2_500
COMMANDS = (
    ("cargo", "fmt", "--all", "--check"),
    (
        "cargo",
        "clippy",
        "--locked",
        "--workspace",
        "--all-targets",
        "--all-features",
        "--",
        "-D",
        "warnings",
    ),
    ("cargo", "test", "--locked", "--workspace", "--all-targets", "--all-features"),
)


def enforce_source_limit() -> None:
    violations = []
    for path in sorted((ROOT / "crates").rglob("*.rs")):
        lines = sum(1 for _ in path.open(encoding="utf-8"))
        if lines > SOURCE_LINE_LIMIT:
            violations.append((path.relative_to(ROOT), lines))
    if violations:
        detail = "\n".join(f"{path}: {lines}" for path, lines in violations)
        raise SystemExit(f"Rust source limit exceeded ({SOURCE_LINE_LIMIT}):\n{detail}")


def main() -> None:
    if len(sys.argv) != 1:
        raise SystemExit("usage: ./check.py")
    enforce_source_limit()
    for command in COMMANDS:
        print("+", " ".join(command), flush=True)
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
