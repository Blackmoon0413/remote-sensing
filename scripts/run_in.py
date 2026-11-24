#!/usr/bin/env python3
"""Run a third-party script with its working directory on the ASCII junction.

Git Bash `cd` resolves D:/rsicc_v2 back to the Chinese project path, which breaks
METEOR (and repos that do sys.path.insert(0, '.')). os.chdir keeps the junction path.

Usage: python run_in.py <workdir> <script.py> [args...]
"""
import os
import runpy
import sys

if __name__ == "__main__":
    workdir, script, *rest = sys.argv[1:]
    os.chdir(workdir)
    assert all(ord(c) < 128 for c in os.getcwd()), f"non-ASCII cwd: {os.getcwd()}"
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    sys.argv = [script] + rest
    runpy.run_path(script, run_name="__main__")
