#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import tomllib
import re
import os
from pathlib import Path
from shutil import copy2, rmtree, which

ROOT = Path(__file__).resolve().parent
WORKSPACE_MANIFEST = ROOT / "Cargo.toml"
SYSTEMD_UNIT_SOURCE = ROOT / "systemd" / "picmash.service"
SYSTEMD_USER_DIR = Path.home() / ".local" / "share" / "systemd" / "user"
DEFAULT_FORMAT = ["cargo", "fmt", "--all", "--check"]
DEFAULT_CLIPPY = ["cargo", "clippy", "--workspace", "--all-targets", "--all-features"]
DEFAULT_TEST = ["cargo", "test", "--workspace", "--all-targets", "--all-features"]
DEFAULT_INSTALL = [
    "cargo",
    "install",
    "--path",
    "crates/picmash-app",
    "--bin",
    "picmash",
    "--locked",
]
FRONTEND_ROOT = ROOT / "apps" / "web"
FRONTEND_ASSET_ROOT = ROOT / "crates" / "picmash-app" / "assets" / "web"
LOCAL_BIN = Path.home() / ".local" / "bin"
ORT_PROVIDER_GLOB = "libonnxruntime_providers*.so"
SOURCE_FILE_EXCLUDES = {"target", ".git"}
FORBIDDEN_JS_SUPPLY_CHAIN_PATHS = (
    FRONTEND_ROOT / "node_modules",
    FRONTEND_ROOT / "package.json",
    FRONTEND_ROOT / "package-lock.json",
    FRONTEND_ROOT / "npm-shrinkwrap.json",
    FRONTEND_ROOT / "pnpm-lock.yaml",
    FRONTEND_ROOT / "yarn.lock",
    FRONTEND_ROOT / "bun.lockb",
    FRONTEND_ROOT / "vite.config.ts",
    FRONTEND_ROOT / "vite.config.js",
)
FRONTEND_SHELL_FORBIDDEN_PATTERNS = (
    r"(?m)(^|[^-\w])\.app-shell\b",
    r"(?m)(^|[^-\w])\.page-body\b",
    r"(?m)(^|[^-\w])\.top-rail\b",
    r"(?m)(^|[^-\w])\.rail-menu\b",
    r"(?m)(^|[^-\w])\.rail\b",
    r"data-page-geometry",
    r"body\.arena-page\b",
    r"body\.board-page\b",
    r"body\.facemash-page\b",
    r"body\.explore-page\b",
    r"body\.triad-page\b",
)


def run(argv: list[str], *, env: dict[str, str] | None = None, cwd: Path = ROOT) -> None:
    print("+", " ".join(argv), flush=True)
    proc = subprocess.run(argv, cwd=cwd, env=env)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def user_systemd_env() -> dict[str, str]:
    env = os.environ.copy()
    runtime_dir = env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    env.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path={runtime_dir}/bus")
    return env


def frontend() -> None:
    if not FRONTEND_ROOT.exists():
        return
    enforce_no_third_party_js()
    if which("tsc") is None:
        raise SystemExit("system TypeScript compiler `tsc` is required")
    run(
        ["cargo", "run", "-q", "-p", "picmash-app", "--bin", "export-ts", "--", "src/generated"],
        cwd=FRONTEND_ROOT,
    )
    rmtree(FRONTEND_ASSET_ROOT, ignore_errors=True)
    run(["tsc", "--project", "tsconfig.json"], cwd=FRONTEND_ROOT)
    rmtree(FRONTEND_ASSET_ROOT / "generated", ignore_errors=True)
    for stale_module in ("contracts.js", "picmash-client.js"):
        (FRONTEND_ASSET_ROOT / stale_module).unlink(missing_ok=True)
    copy2(FRONTEND_ROOT / "src" / "styles.css", FRONTEND_ASSET_ROOT / "picmash-client.css")


def workspace_metadata() -> dict[str, object]:
    workspace = tomllib.loads(WORKSPACE_MANIFEST.read_text(encoding="utf-8"))
    metadata = workspace.get("workspace", {}).get("metadata", {})
    rust_starter = metadata.get("rust-starter", {})
    return rust_starter if isinstance(rust_starter, dict) else {}


def configured_source_file_max_lines() -> int | None:
    source_files = workspace_metadata().get("source_files")
    if not isinstance(source_files, dict):
        return None
    max_lines = source_files.get("max_lines")
    if not isinstance(max_lines, int) or max_lines <= 0:
        return None
    return max_lines


def iter_rust_source_files() -> list[Path]:
    paths: list[Path] = []
    for path in ROOT.rglob("*.rs"):
        relative = path.relative_to(ROOT)
        if any(part in SOURCE_FILE_EXCLUDES for part in relative.parts):
            continue
        paths.append(path)
    return sorted(paths)


def enforce_source_file_cap() -> None:
    max_lines = configured_source_file_max_lines()
    if max_lines is None:
        return
    violations: list[tuple[int, Path]] = []
    for path in iter_rust_source_files():
        with path.open(encoding="utf-8") as handle:
            line_count = sum(1 for _ in handle)
        if line_count > max_lines:
            violations.append((line_count, path.relative_to(ROOT)))
    if not violations:
        return
    print(f"rust source file cap exceeded ({max_lines} lines):", file=sys.stderr)
    for line_count, path in violations:
        print(f"  {line_count:>5} {path}", file=sys.stderr)
    raise SystemExit(1)


def enforce_stylesheet_ownership() -> None:
    stylesheet = FRONTEND_ROOT / "src" / "styles.css"
    if not stylesheet.exists():
        return
    source = stylesheet.read_text(encoding="utf-8")
    violations = [
        pattern
        for pattern in FRONTEND_SHELL_FORBIDDEN_PATTERNS
        if re.search(pattern, source) is not None
    ]
    if not violations:
        return
    print("frontend stylesheet trespasses on shared shell selectors:", file=sys.stderr)
    for fragment in violations:
        print(f"  {fragment}", file=sys.stderr)
    raise SystemExit(1)


def enforce_no_third_party_js() -> None:
    violations = [path for path in FORBIDDEN_JS_SUPPLY_CHAIN_PATHS if path.exists()]
    if not violations:
        return
    print("third-party JavaScript supply-chain surfaces are forbidden:", file=sys.stderr)
    for path in violations:
        print(f"  {path.relative_to(ROOT)}", file=sys.stderr)
    raise SystemExit(1)


def configured_command(name: str, default: list[str]) -> list[str]:
    configured = workspace_metadata().get(name)
    if not (
        isinstance(configured, list)
        and configured
        and all(isinstance(part, str) for part in configured)
    ):
        return default
    return list(configured)


def normalized_clippy_command() -> list[str]:
    configured = configured_command("clippy_command", DEFAULT_CLIPPY)
    command: list[str] = []
    skip_next = False
    for index, part in enumerate(configured):
        if skip_next:
            skip_next = False
            continue
        if part.startswith("--message-format="):
            continue
        if part == "--message-format" and index + 1 < len(configured):
            skip_next = True
            continue
        command.append(part)
    return command or DEFAULT_CLIPPY


def normalized_install_command() -> list[str]:
    configured = configured_command("install_command", DEFAULT_INSTALL)
    command = list(configured)
    if not any(part == "--root" or part.startswith("--root=") for part in command):
        command.extend(["--root", str(Path.home() / ".local")])
    if "--force" not in command:
        command.append("--force")
    return command


def check() -> None:
    enforce_source_file_cap()
    enforce_stylesheet_ownership()
    enforce_no_third_party_js()
    frontend()
    run(configured_command("format_command", DEFAULT_FORMAT))
    run(normalized_clippy_command())
    run(configured_command("test_command", DEFAULT_TEST))


def install() -> None:
    enforce_source_file_cap()
    enforce_stylesheet_ownership()
    enforce_no_third_party_js()
    frontend()
    run(normalized_install_command())
    install_ort_provider_dylibs()
    install_user_service()


def install_ort_provider_dylibs() -> None:
    release_root = ROOT / "target" / "release"
    if not release_root.exists():
        return
    dylibs = sorted(release_root.glob(ORT_PROVIDER_GLOB))
    if not dylibs:
        return
    LOCAL_BIN.mkdir(parents=True, exist_ok=True)
    for dylib in dylibs:
        copy2(dylib, LOCAL_BIN / dylib.name)


def install_user_service() -> None:
    if not SYSTEMD_UNIT_SOURCE.exists():
        return
    SYSTEMD_USER_DIR.mkdir(parents=True, exist_ok=True)
    unit_target = SYSTEMD_USER_DIR / "picmash.service"
    source = SYSTEMD_UNIT_SOURCE.read_text(encoding="utf-8")
    if not unit_target.exists() or unit_target.read_text(encoding="utf-8") != source:
        unit_target.write_text(source, encoding="utf-8")
    env = user_systemd_env()
    run(["systemctl", "--user", "daemon-reload"], env=env)
    run(["systemctl", "--user", "enable", "picmash.service"], env=env)
    run(["systemctl", "--user", "try-restart", "picmash.service"], env=env)


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "check"
    if len(sys.argv) > 2:
        raise SystemExit("usage: ./check.py [check|install]")
    if command == "check":
        check()
        return
    if command == "install":
        install()
        return
    raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
