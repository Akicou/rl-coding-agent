"""Rust sandbox executor."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from rl_agent.languages.base import ExecutionResult, LanguageExecutor

_KNOWN_CRATES = {
    "serde": 'serde = { version = "1", features = ["derive"] }',
    "serde_json": 'serde_json = "1"',
    "regex": 'regex = "1"',
    "itertools": 'itertools = "0.13"',
    "num": 'num = "0.4"',
    "rand": 'rand = "0.8"',
}


class RustExecutor(LanguageExecutor):
    """Execute Rust code in a temporary Cargo project."""

    def extract_deps(self, code: str) -> set[str]:
        """Extract supported external crates from use and extern statements."""

        deps = set(re.findall(r"^\s*use\s+([a-zA-Z_][\w]*)", code, re.MULTILINE))
        deps.update(
            re.findall(r"^\s*extern\s+crate\s+([a-zA-Z_][\w]*)", code, re.MULTILINE)
        )
        return deps & _KNOWN_CRATES.keys()

    def execute(self, code: str, stdin: str = "", timeout: int = 10) -> ExecutionResult:
        """Build and run a Rust binary crate."""

        with tempfile.TemporaryDirectory(prefix="tmp_rust_") as tmpdir:
            init_result = self._run(
                ["cargo", "init", "--bin", "--name", "agent", "."],
                cwd=tmpdir,
                timeout=300,
            )
            if not init_result.success:
                return init_result
            deps = self.extract_deps(code)
            cargo_toml = [
                "[package]",
                'name = "agent"',
                'version = "0.1.0"',
                'edition = "2021"',
                "",
                "[dependencies]",
            ]
            for dep in sorted(deps):
                cargo_toml.append(_KNOWN_CRATES[dep])
            Path(tmpdir, "Cargo.toml").write_text(
                "\n".join(cargo_toml) + "\n", encoding="utf-8"
            )
            Path(tmpdir, "src", "main.rs").write_text(code, encoding="utf-8")
            build_result = self._run(
                ["cargo", "build", "--release"], cwd=tmpdir, timeout=300
            )
            if not build_result.success:
                return build_result
            binary = Path(tmpdir, "target", "release", "agent")
            return self._run([str(binary)], cwd=tmpdir, stdin=stdin, timeout=timeout)
