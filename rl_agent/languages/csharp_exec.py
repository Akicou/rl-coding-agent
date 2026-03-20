"""C# sandbox executor."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from rl_agent.languages.base import ExecutionResult, LanguageExecutor

_NUGET_MAP = {
    "Newtonsoft.Json": "Newtonsoft.Json",
    "CsvHelper": "CsvHelper",
    "Dapper": "Dapper",
    "Npgsql": "Npgsql",
    "Spectre.Console": "Spectre.Console",
}

_CSPROJ = """<Project Sdk=\"Microsoft.NET.Sdk\">\n  <PropertyGroup>\n    <OutputType>Exe</OutputType>\n    <TargetFramework>net8.0</TargetFramework>\n    <ImplicitUsings>enable</ImplicitUsings>\n    <Nullable>enable</Nullable>\n    <AllowUnsafeBlocks>true</AllowUnsafeBlocks>\n  </PropertyGroup>\n</Project>\n"""


class CSharpExecutor(LanguageExecutor):
    """Execute C# programs with a temporary project file."""

    def extract_deps(self, code: str) -> set[str]:
        """Extract namespaces that map to NuGet packages."""

        namespaces = set(re.findall(r"^\s*using\s+([\w.]+);", code, re.MULTILINE))
        return {
            package
            for namespace, package in _NUGET_MAP.items()
            if namespace in namespaces
        }

    def execute(self, code: str, stdin: str = "", timeout: int = 10) -> ExecutionResult:
        """Execute a C# project with optional NuGet restore."""

        with tempfile.TemporaryDirectory(prefix="tmp_csharp_") as tmpdir:
            root = Path(tmpdir)
            (root / "agent.csproj").write_text(_CSPROJ, encoding="utf-8")
            (root / "Program.cs").write_text(code, encoding="utf-8")
            deps = self.extract_deps(code)
            for dep in sorted(deps):
                add_result = self._run(
                    ["dotnet", "add", "package", dep], cwd=tmpdir, timeout=300
                )
                if not add_result.success:
                    return add_result
            restore_result = self._run(["dotnet", "restore"], cwd=tmpdir, timeout=300)
            if not restore_result.success:
                return restore_result
            return self._run(
                ["dotnet", "run", "--project", "agent.csproj"],
                cwd=tmpdir,
                stdin=stdin,
                timeout=timeout,
            )
