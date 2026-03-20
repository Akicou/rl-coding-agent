"""Python sandbox executor."""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

from rl_agent.languages.base import ExecutionResult, LanguageExecutor

_ALIASES = {
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "yaml": "PyYAML",
    "bs4": "beautifulsoup4",
    "sklearn": "scikit-learn",
}
_STDLIB = set(getattr(sys, "stdlib_module_names", set())) | {
    "collections",
    "math",
    "heapq",
    "itertools",
    "functools",
    "typing",
    "pathlib",
    "json",
    "re",
    "sys",
    "os",
    "statistics",
    "bisect",
    "string",
    "random",
    "time",
    "datetime",
}


class PythonExecutor(LanguageExecutor):
    """Execute Python source files in a temporary directory."""

    def extract_deps(self, code: str) -> set[str]:
        """Extract third-party top-level imports."""

        deps: set[str] = set()
        for match in re.finditer(r"^\s*import\s+([\w.,\s]+)", code, re.MULTILINE):
            for item in match.group(1).split(","):
                root = item.strip().split(" as ")[0].split(".")[0]
                if root and root not in _STDLIB:
                    deps.add(_ALIASES.get(root, root))
        for match in re.finditer(
            r"^\s*from\s+([\w.]+)\s+import\s+", code, re.MULTILINE
        ):
            root = match.group(1).split(".")[0]
            if root and root not in _STDLIB:
                deps.add(_ALIASES.get(root, root))
        return deps

    def _install(self, deps: set[str]) -> None:
        """Install missing Python packages."""

        if deps:
            self._run(
                [sys.executable, "-m", "pip", "install", *sorted(deps)], timeout=300
            )

    def execute(self, code: str, stdin: str = "", timeout: int = 10) -> ExecutionResult:
        """Execute Python code after ensuring dependencies."""

        self.ensure_deps(self.extract_deps(code))
        with tempfile.TemporaryDirectory(prefix="tmp_python_") as tmpdir:
            path = Path(tmpdir) / "solution.py"
            path.write_text(code, encoding="utf-8")
            return self._run(
                [sys.executable, str(path)], cwd=tmpdir, stdin=stdin, timeout=timeout
            )
