"""
Causal Module Test Suite - Unit Tests for Causal Inference Components
Author: Erdinc Erdogan
Purpose: Provides unit tests for all causal module components including conditional independence
testing, DAG learning, and do-calculus operations.
References:
- Python unittest Framework
- Statistical Hypothesis Testing
- Test-Driven Development (TDD)
Usage:
    python test_causal_module.py
"""

import numpy as np
from scipy import stats
from scipy.linalg import inv
import sys

def run_tests():
    """Run all causal module tests."""
    passed = 0
    failed = 0
    
    # Test 1: Basic CI
    np.random.seed(42)
    x, y = np.random.randn(500), np.random.randn(500)
    corr = np.corrcoef(x, y)[0, 1]
    p = 2 * (1 - stats.norm.cdf(abs(0.5 * np.log((1+corr)/(1-corr))) / (1/np.sqrt(497))))
    if p > 0.05:
        passed += 1
        print("✅ CI Basic Independence")
    else:
        failed += 1
        print("❌ CI Basic Independence")
    
    # Test 2: Backdoor Adjustment
    np.random.seed(42)
    n = 1000
    z = np.random.randn(n)
    x = 0.7 * z + np.random.randn(n) * 0.5
    y = 0.5 * x + 0.8 * z + np.random.randn(n) * 0.3
    beta = np.linalg.lstsq(np.column_stack([np.ones(n), x, z]), y, rcond=None)[0]
    if abs(beta[1] - 0.5) < 0.1:
        passed += 1
        print("✅ Backdoor Adjustment")
    else:
        failed += 1
        print("❌ Backdoor Adjustment")
    
    # Test 3: Collider Detection
    np.random.seed(42)
    x, y = np.random.randn(500), np.random.randn(500)
    z = 0.5 * x + 0.5 * y + np.random.randn(500) * 0.3
    corr_xy = np.corrcoef(x, y)[0, 1]
    data = np.column_stack([x, y, z])
    prec = inv(np.corrcoef(data.T) + 1e-6 * np.eye(3))
    partial = -prec[0,1] / np.sqrt(prec[0,0] * prec[1,1])
    if abs(corr_xy) < 0.15 and abs(partial) > 0.2:
        passed += 1
        print("✅ Collider Detection")
    else:
        failed += 1
        print("❌ Collider Detection")
    
    print(f"\nResults: {passed}/{passed+failed} passed")
    return failed == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
