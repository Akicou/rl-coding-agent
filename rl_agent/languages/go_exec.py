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

        with tempfile.TemporaryDirectory(prefix="tmp_go_") as tmpdir:
            root = Path(tmpdir)
            (root / "main.go").write_text(code, encoding="utf-8")
            init_result = self._run(
                ["go", "mod", "init", "agent"], cwd=tmpdir, timeout=timeout
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
            return self._run(
                ["go", "run", "main.go"], cwd=tmpdir, stdin=stdin, timeout=timeout
            )
