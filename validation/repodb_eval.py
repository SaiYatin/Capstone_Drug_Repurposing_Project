"""
RepoDB-Style Evaluation
========================

Evaluates the model on the test set which serves as our RepoDB-equivalent
benchmark. The test set contains held-out diseases with both positive
(known drug-disease) and negative (random) pairs.

This is effectively the same as validation/metrics.py but structured
to match the RepoDB benchmark format described in the architecture doc.

Usage:
    python validation/repodb_eval.py

Author: Yatin
Date: April 2026
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# RepoDB evaluation is handled by validation/metrics.py
# This module provides a convenience wrapper

from validation.metrics import main as run_metrics

if __name__ == '__main__':
    print("RepoDB-style evaluation (via comprehensive metrics)...")
    print("=" * 80)
    run_metrics()
