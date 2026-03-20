"""Node.js sandbox executor."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

from rl_agent.languages.base import ExecutionResult, LanguageExecutor

_NODE_STDLIB = {
    "assert",
    "buffer",
    "child_process",
    "crypto",
    "events",
    "fs",
    "http",
    "https",
    "net",
    "os",
    "path",
    "readline",
    "stream",
    "timers",
    "url",
    "util",
    "zlib",
}


class NodeExecutor(LanguageExecutor):
    """Execute Node.js code in a temporary npm project."""

    def extract_deps(self, code: str) -> set[str]:
        """Extract package imports from CommonJS and ESM syntax."""

        deps = set(re.findall(r"require\(['\"]([^'\"]+)['\"]\)", code))
        deps.update(re.findall(r"from\s+['\"]([^'\"]+)['\"]", code))
        deps.update(re.findall(r"import\s+['\"]([^'\"]+)['\"]", code))
        clean: set[str] = set()
        for dep in deps:
            if dep.startswith(".") or dep.startswith("node:"):
                continue
            package = (
                dep.split("/")[0]
                if not dep.startswith("@")
                else "/".join(dep.split("/")[:2])
            )
            if package not in _NODE_STDLIB:
                clean.add(package)
        return clean

    def execute(self, code: str, stdin: str = "", timeout: int = 10) -> ExecutionResult:
        """Execute Node.js code after creating package metadata."""

        with tempfile.TemporaryDirectory(prefix="tmp_node_") as tmpdir:
            root = Path(tmpdir)
            package_json = {
                "name": "agent",
                "version": "1.0.0",
                "private": True,
                "type": "commonjs",
            }
            (root / "package.json").write_text(
                json.dumps(package_json, indent=2), encoding="utf-8"
            )
            (root / "solution.js").write_text(code, encoding="utf-8")
            deps = self.extract_deps(code)
            if deps:
                install_result = self._run(
                    ["npm", "install", *sorted(deps)], cwd=tmpdir, timeout=300
                )
                if not install_result.success:
                    return install_result
            return self._run(
                ["node", "solution.js"], cwd=tmpdir, stdin=stdin, timeout=timeout
            )
