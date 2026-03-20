"""Tests for language executor sandboxes."""

from __future__ import annotations

import shutil

import pytest

from rl_agent.languages.cpp_exec import CppExecutor
from rl_agent.languages.csharp_exec import CSharpExecutor
from rl_agent.languages.go_exec import GoExecutor
from rl_agent.languages.node_exec import NodeExecutor
from rl_agent.languages.python_exec import PythonExecutor
from rl_agent.languages.rust_exec import RustExecutor


PY_HELLO = 'print("Hello World")\n'
PY_BAD = 'print("oops"\n'
PY_LOOP = "while True:\n    pass\n"

GO_HELLO = 'package main\nimport "fmt"\nfunc main() { fmt.Println("Hello World") }\n'
GO_BAD = "package main\nfunc main( { }\n"
GO_LOOP = "package main\nfunc main() { for {} }\n"

NODE_HELLO = 'console.log("Hello World")\n'
NODE_BAD = 'console.log("Hello World"\n'
NODE_LOOP = "while (true) {}\n"

CS_HELLO = 'using System;\nConsole.WriteLine("Hello World");\n'
CS_BAD = 'using System;\nConsole.WriteLine("Hello World"\n'
CS_LOOP = "while (true) { }\n"

CPP_HELLO = '#include <iostream>\nint main(){ std::cout << "Hello World\\n"; }\n'
CPP_BAD = "#include <iostream>\nint main( { return 0; }\n"
CPP_LOOP = "#include <iostream>\nint main(){ while(true){} }\n"

RUST_HELLO = 'fn main() { println!("Hello World"); }\n'
RUST_BAD = 'fn main( { println!("Hello World"); }\n'
RUST_LOOP = "fn main() { loop {} }\n"


def test_python_executor_hello() -> None:
    result = PythonExecutor().execute(PY_HELLO)
    assert result.success
    assert result.stdout.strip() == "Hello World"


def test_python_executor_failure() -> None:
    assert not PythonExecutor().execute(PY_BAD).success


def test_python_timeout() -> None:
    result = PythonExecutor().execute(PY_LOOP, timeout=1)
    assert result.timed_out


@pytest.mark.skipif(shutil.which("go") is None, reason="go not installed")
def test_go_executor_hello() -> None:
    result = GoExecutor().execute(GO_HELLO)
    assert result.success
    assert result.stdout.strip() == "Hello World"


@pytest.mark.skipif(shutil.which("go") is None, reason="go not installed")
def test_go_executor_failure() -> None:
    assert not GoExecutor().execute(GO_BAD).success


@pytest.mark.skipif(shutil.which("go") is None, reason="go not installed")
def test_go_timeout() -> None:
    result = GoExecutor().execute(GO_LOOP, timeout=1)
    assert result.timed_out


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_node_executor_hello() -> None:
    result = NodeExecutor().execute(NODE_HELLO)
    assert result.success
    assert result.stdout.strip() == "Hello World"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_node_executor_failure() -> None:
    assert not NodeExecutor().execute(NODE_BAD).success


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_node_timeout() -> None:
    result = NodeExecutor().execute(NODE_LOOP, timeout=1)
    assert result.timed_out


@pytest.mark.skipif(shutil.which("dotnet") is None, reason="dotnet not installed")
def test_csharp_executor_hello() -> None:
    result = CSharpExecutor().execute(CS_HELLO)
    assert result.success
    assert result.stdout.strip() == "Hello World"


@pytest.mark.skipif(shutil.which("dotnet") is None, reason="dotnet not installed")
def test_csharp_executor_failure() -> None:
    assert not CSharpExecutor().execute(CS_BAD).success


@pytest.mark.skipif(shutil.which("dotnet") is None, reason="dotnet not installed")
def test_csharp_timeout() -> None:
    result = CSharpExecutor().execute(CS_LOOP, timeout=1)
    assert result.timed_out


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ not installed")
def test_cpp_executor_hello() -> None:
    result = CppExecutor().execute(CPP_HELLO)
    assert result.success
    assert result.stdout.strip() == "Hello World"


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ not installed")
def test_cpp_executor_failure() -> None:
    assert not CppExecutor().execute(CPP_BAD).success


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ not installed")
def test_cpp_timeout() -> None:
    result = CppExecutor().execute(CPP_LOOP, timeout=1)
    assert result.timed_out


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo not installed")
def test_rust_executor_hello() -> None:
    result = RustExecutor().execute(RUST_HELLO)
    assert result.success
    assert result.stdout.strip() == "Hello World"


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo not installed")
def test_rust_executor_failure() -> None:
    assert not RustExecutor().execute(RUST_BAD).success


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo not installed")
def test_rust_timeout() -> None:
    result = RustExecutor().execute(RUST_LOOP, timeout=1)
    assert result.timed_out
