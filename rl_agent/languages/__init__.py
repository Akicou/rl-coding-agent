"""Language registry for sandbox execution."""

from __future__ import annotations

from rl_agent.languages.base import LanguageProfile
from rl_agent.languages.cpp_exec import CppExecutor
from rl_agent.languages.csharp_exec import CSharpExecutor
from rl_agent.languages.go_exec import GoExecutor
from rl_agent.languages.node_exec import NodeExecutor
from rl_agent.languages.python_exec import PythonExecutor
from rl_agent.languages.rust_exec import RustExecutor

LANGUAGE_REGISTRY: dict[str, LanguageProfile] = {
    "python": LanguageProfile(
        name="Python",
        fence="python",
        executor=PythonExecutor(),
        io_hint="Read stdin and print exact stdout.",
    ),
    "golang": LanguageProfile(
        name="Go",
        fence="go",
        executor=GoExecutor(),
        io_hint="Use package main and fmt.Println for output.",
    ),
    "nodejs": LanguageProfile(
        name="Node.js",
        fence="javascript",
        executor=NodeExecutor(),
        io_hint="Use process.stdin and console.log.",
    ),
    "csharp": LanguageProfile(
        name="C#",
        fence="csharp",
        executor=CSharpExecutor(),
        io_hint="Use Console.ReadLine and Console.WriteLine.",
    ),
    "cpp": LanguageProfile(
        name="C++",
        fence="cpp",
        executor=CppExecutor(),
        io_hint="Use std::cin and std::cout.",
    ),
    "rust": LanguageProfile(
        name="Rust",
        fence="rust",
        executor=RustExecutor(),
        io_hint="Use std::io for stdin and stdout.",
    ),
}
