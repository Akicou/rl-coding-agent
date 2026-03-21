"""Diagnostic script to verify all language runtimes are correctly installed and working."""

import os
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_agent.languages import LANGUAGE_REGISTRY

# Test cases for each language
TESTS = {
    "python": "print('hello from python')",
    "golang": 'package main\nimport "fmt"\nfunc main() { fmt.Println("hello from go") }',
    "nodejs": "console.log('hello from node');",
    "csharp": 'using System;\nclass Program {\n  static void Main() {\n    Console.WriteLine("hello from csharp");\n  }\n}',
    "cpp": '#include <iostream>\nint main() { std::cout << "hello from cpp" << std::endl; return 0; }',
    "rust": 'fn main() { println!("hello from rust"); }',
}


def check_runtimes():
    """Iterate through the registry and run a smoke test for each language."""

    print("🔍 Checking Language Runtimes...\n")
    results = {}

    for key, profile in LANGUAGE_REGISTRY.items():
        print(f"--- Testing {profile.name} ({key}) ---")
        code = TESTS.get(key)
        if not code:
            print(f"❌ No test case defined for {key}\n")
            results[key] = False
            continue

        try:
            # We use a 30s timeout for build-heavy languages like Rust
            result = profile.executor.execute(code, timeout=60)
            if result.success:
                print(f"✅ Output: {result.stdout.strip()}")
                results[key] = True
            else:
                print(f"❌ Execution Failed!")
                print(f"Stderr: {result.stderr.strip()}")
                results[key] = False
        except Exception as exc:
            print(f"💥 Exception during check: {exc}")
            results[key] = False
        print()

    print("=" * 30)
    print("Summary:")
    all_ok = True
    for key, ok in results.items():
        status = "✅ OK" if ok else "❌ FAILED"
        print(f"{key:10}: {status}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n✨ All runtimes are correctly configured!")
        sys.exit(0)
    else:
        print("\n⚠️ Some runtimes are missing or broken. Please check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    check_runtimes()
