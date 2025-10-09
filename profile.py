"""Launcher for the heavy profiler implementation.

This script spawns a fresh Python process to run `profile_impl.py` using the
same interpreter (sys.executable). Doing this avoids import-time collisions
between the local filename `profile.py` and the stdlib `profile` module when
heavy libraries (torch_geometric / torch._dynamo) import cProfile/profile.
"""
import sys
import subprocess
from pathlib import Path


def main():
    impl = Path(__file__).with_name('profile_impl.py')
    if not impl.exists():
        print(f"profile_impl.py not found at {impl}")
        return 2
    cmd = [sys.executable, str(impl)]
    return subprocess.call(cmd)


if __name__ == '__main__':
    raise SystemExit(main())