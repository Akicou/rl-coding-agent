"""Shared abstractions for language execution sandboxes."""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ExecutionResult:
    """Captured subprocess result for a code execution."""

    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False

    @property
    def success(self) -> bool:
        """Return whether execution completed successfully."""

        return not self.timed_out and self.returncode == 0


@dataclass(slots=True)
class LanguageProfile:
    """Metadata used for prompting and runtime lookup."""

    name: str
    fence: str
    executor: "LanguageExecutor"
    io_hint: str


class LanguageExecutor(ABC):
    """Abstract interface for a language-specific sandbox executor."""

    _cache: set[str] = set()

    @abstractmethod
    def execute(self, code: str, stdin: str = "", timeout: int = 10) -> ExecutionResult:
        """Execute source code with optional stdin."""

    @abstractmethod
    def extract_deps(self, code: str) -> set[str]:
        """Extract runtime dependencies from source code."""

    def ensure_deps(self, deps: set[str]) -> None:
        """Install dependencies only once per process."""

        pending = deps - self._cache
        if not pending:
            return
        self._install(pending)
        self._cache.update(pending)

    def _install(self, deps: set[str]) -> None:
        """Install dependencies for this runtime."""

    def _run(
        self,
        command: list[str],
        cwd: str | None = None,
        stdin: str = "",
        timeout: int = 10,
    ) -> ExecutionResult:
        """Run a subprocess command with timeout handling."""

        import shutil
        import sys

        # On Windows, we need to find the correct executable (e.g., .cmd or .exe)
        # shutil.which handles this correctly by checking PATHEXT.
        cmd = list(command)
        if not Path(cmd[0]).is_absolute():
            exe = shutil.which(cmd[0])
            if exe:
                cmd[0] = exe

        try:
            proc = subprocess.run(
                cmd,
                input=stdin,
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=timeout,
                check=False,
                shell=False,
            )
            return ExecutionResult(
                stdout=proc.stdout,
                stderr=proc.stderr,
                returncode=proc.returncode,
                timed_out=False,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                returncode=124,
                timed_out=True,
            )
        except FileNotFoundError:
            # Fallback for some Windows environments where shell=True is needed for scripts
            if sys.platform == "win32":
                try:
                    proc = subprocess.run(
                        cmd,
                        input=stdin,
                        capture_output=True,
                        text=True,
                        cwd=cwd,
                        timeout=timeout,
                        check=False,
                        shell=True,
                    )
                    return ExecutionResult(
                        stdout=proc.stdout,
                        stderr=proc.stderr,
                        returncode=proc.returncode,
                        timed_out=False,
                    )
                except Exception as exc2:
                    return ExecutionResult(
                        stdout="",
                        stderr=str(exc2),
                        returncode=1,
                        timed_out=False,
                    )
            return ExecutionResult(
                stdout="",
                stderr=f"Executable not found: {cmd[0]}",
                returncode=1,
                timed_out=False,
            )
