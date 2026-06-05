#!/usr/bin/env python3
"""Run modality ablation analysis (all 4 regime/horizon configs)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    from evaluation.ablation import run_ablation_suite

    ablation_configs = [
        ("baseline_multimodal", "V06"),
        ("baseline_multimodal", "V08"),
        ("baseline_plus_12m", "V08"),
        ("rolling", "V08"),
    ]
    for regime, horizon in ablation_configs:
        try:
            run_ablation_suite(regime=regime, horizon=horizon)
        except Exception as e:
            print(f"WARNING: Ablation {regime}/{horizon} failed: {e}")
