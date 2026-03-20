"""C++ sandbox executor."""

from __future__ import annotations

import tempfile
from pathlib import Path

from rl_agent.languages.base import ExecutionResult, LanguageExecutor


class CppExecutor(LanguageExecutor):
    """Compile and execute C++17 programs."""

    def extract_deps(self, code: str) -> set[str]:
        """C++ executor only supports standard library dependencies."""

        return set()

    def execute(self, code: str, stdin: str = "", timeout: int = 10) -> ExecutionResult:
        """Compile and run a C++ source file."""

        with tempfile.TemporaryDirectory(prefix="tmp_cpp_") as tmpdir:
            root = Path(tmpdir)
            (root / "main.cpp").write_text(code, encoding="utf-8")
            binary = root / "main"
            compile_result = self._run(
                ["g++", "-O2", "-std=c++17", "main.cpp", "-o", str(binary)],
                cwd=tmpdir,
                timeout=timeout,
            )
            if not compile_result.success:
                return compile_result
            return self._run([str(binary)], cwd=tmpdir, stdin=stdin, timeout=timeout)
