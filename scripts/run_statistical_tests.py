#!/usr/bin/env python3
"""Run post-hoc statistical tests with FDR correction."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.statistical_tests import run_all_statistical_tests

if __name__ == "__main__":
    run_all_statistical_tests()
