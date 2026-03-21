"""Go sandbox executor."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from rl_agent.languages.base import ExecutionResult, LanguageExecutor


class GoExecutor(LanguageExecutor):
    """Execute Go code inside a fresh module."""

    def extract_deps(self, code: str) -> set[str]:
        """Extract non-stdlib import paths."""

        imports = set(re.findall(r'"([^"]+)"', code))
        return {item for item in imports if "." in item.split("/")[0]}

    def execute(self, code: str, stdin: str = "", timeout: int = 10) -> ExecutionResult:
        """Execute Go code after creating a temporary module."""

        import sys

        with tempfile.TemporaryDirectory(prefix="tmp_go_") as tmpdir:
            root = Path(tmpdir)
            (root / "main.go").write_text(code, encoding="utf-8")
            init_result = self._run(
                ["go", "mod", "init", "agent"], cwd=tmpdir, timeout=60
            )
            if not init_result.success:
                return init_result
            deps = self.extract_deps(code)
            if deps:
                get_result = self._run(
                    ["go", "get", *sorted(deps)], cwd=tmpdir, timeout=300
                )
                if not get_result.success:
                    return get_result
            binary_name = "agent.exe" if sys.platform == "win32" else "agent"
            build_result = self._run(
                ["go", "build", "-o", binary_name, "main.go"],
                cwd=tmpdir,
                timeout=max(timeout, 60),
            )
            if not build_result.success:
                return build_result
            return self._run(
                [str(root / binary_name)],
                cwd=tmpdir,
                stdin=stdin,
                timeout=timeout,
            )
