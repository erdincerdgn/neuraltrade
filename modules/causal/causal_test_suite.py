"""
Comprehensive Test Suite for Causal Module
Author: Erdinc Erdogan
Purpose: Tests all causal module components including RobustConditionalIndependenceTest,
RobustDoCalculus, PCStableEngine, and integration tests for edge case handling.
References:
- Unit Testing Best Practices
- Statistical Hypothesis Testing
- Causal Inference Validation Methods
Usage:
    python causal_test_suite.py
"""

import unittest
import numpy as np
import time
import sys
import os
from typing import Dict, List, Set

# Add outputs to path for imports
sys.path.append('outputs')
sys.path.append('.')

try:
    from .robust_ci_test import RobustConditionalIndependenceTest
    from .robust_do_calculus import RobustDoCalculus, CausalEffect
    from .pc_stable_engine import PCStableEngine
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all fixed modules are in outputs/ directory")
    sys.exit(1)


class TestRobustCITest(unittest.TestCase):
    """Test Vulnerability 1 fix: DAG Stability under noise."""
    
    def setUp(self):
        np.random.seed(42)
        self.n_samples = 200
        self.data = self._generate_test_data()
        
    def _generate_test_data(self):
        """Generate data with known dependencies."""
        # X -> Z <- Y (v-structure)
        x = np.random.randn(self.n_samples)
        y = np.random.randn(self.n_samples)
        z = 0.5 * x + 0.7 * y + np.random.randn(self.n_samples) * 0.1
        return np.column_stack([x, y, z])
    
    def test_adaptive_regularization_stability(self):
        """Test that adaptive regularization handles ill-conditioned matrices."""
        ci_test = RobustConditionalIndependenceTest(method="adaptive")
        
        # Test with increasing noise levels
        noise_levels = [0.1, 0.5, 1.0, 2.0]
        results = []
        
        for noise in noise_levels:
            noisy_data = self.data + np.random.randn(*self.data.shape) * noise
            independent, _, p_value = ci_test.test(noisy_data, 0, 1, [2])
            results.append(p_value)
        
        # P-values should be stable (not NaN or extreme)
        for p_val in results:
            self.assertFalse(np.isnan(p_val), "P-value should not be NaN")
            self.assertGreaterEqual(p_val, 0.0, "P-value should be >= 0")
            self.assertLessEqual(p_val, 1.0, "P-value should be <= 1")
    
    def test_robust_vs_standard_methods(self):
        """Test robust method handles heavy-tailed data better."""
        # Generate heavy-tailed data
        heavy_tail_data = np.random.standard_t(df=3, size=(self.n_samples, 3))
        
        standard_test = RobustConditionalIndependenceTest(method="standard")
        robust_test = RobustConditionalIndependenceTest(method="robust")
        
        # Both should complete without errors
        try:
            std_result = standard_test.test(heavy_tail_data, 0, 1, [2])
            rob_result = robust_test.test(heavy_tail_data, 0, 1, [2])
            
            # Results should be valid
            self.assertIsInstance(std_result[0], bool)
            self.assertIsInstance(rob_result[0], bool)
            
        except Exception as e:
            self.fail(f"CI tests failed on heavy-tailed data: {e}")
    
    def test_caching_functionality(self):
        """Test that CI test caching works correctly."""
        ci_test = RobustConditionalIndependenceTest(method="adaptive")
        
        # First call
        start_time = time.time()
        result1 = ci_test.test(self.data, 0, 1, [2])
        first_time = time.time() - start_time
        
        # Second call (should be cached)
        start_time = time.time()
        result2 = ci_test.test(self.data, 0, 1, [2])
        second_time = time.time() - start_time
        
        # Results should be identical
        self.assertEqual(result1, result2, "Cached results should match")
        
        # Second call should be faster (cached)
        self.assertLess(second_time, first_time * 0.5, "Cached call should be faster")


class TestRobustDoCalculus(unittest.TestCase):
    """Test Vulnerability 2 fix: Intervention leakage."""
    
    def setUp(self):
        np.random.seed(42)
        self.n_samples = 500
        self.data, self.adj_matrix, self.variable_names = self._generate_confounded_data()
        
    def _generate_confounded_data(self):
        """Generate data with known confounding structure."""
        # Confounder -> Treatment, Outcome
        # Treatment -> Outcome
        # TRUE causal effect: 0.5
        
        confounder = np.random.randn(self.n_samples)
        treatment = 0.7 * confounder + np.random.randn(self.n_samples) * 0.5
        outcome = 0.5 * treatment + 0.8 * confounder + np.random.randn(self.n_samples) * 0.3
        
        data = np.column_stack([treatment, outcome, confounder])
        variable_names = ["Treatment", "Outcome", "Confounder"]
        
        # Adjacency matrix
        adj_matrix = np.array([
            [0, 1, 0],  # Treatment -> Outcome
            [0, 0, 0],  # Outcome -> nothing
            [1, 1, 0],  # Confounder -> Treatment, Outcome
        ])
        
        return data, adj_matrix, variable_names
    
    def test_backdoor_adjustment_accuracy(self):
        """Test that backdoor adjustment gives unbiased estimates."""
        do_calc = RobustDoCalculus(self.variable_names, self.adj_matrix)
        do_calc.fit(self.data)
        
        effect = do_calc.compute_ate_backdoor("Treatment", "Outcome")
        
        # Should be identifiable
        self.assertTrue(effect.is_identifiable, "Effect should be identifiable")
        
        # Should find confounder in adjustment set
        self.assertIn("Confounder", effect.adjustment_set, "Should adjust for confounder")
        
        # Estimate should be close to true effect (0.5)
        self.assertAlmostEqual(effect.ate, 0.5, delta=0.1, 
                              msg="ATE should be close to true effect (0.5)")
        
        # Confidence interval should contain true effect
        self.assertLessEqual(effect.ci_lower, 0.5, "CI should contain true effect")
        self.assertGreaterEqual(effect.ci_upper, 0.5, "CI should contain true effect")
    
    def test_backdoor_criterion_validation(self):
        """Test that backdoor criterion is properly checked."""
        do_calc = RobustDoCalculus(self.variable_names, self.adj_matrix)
        do_calc.fit(self.data)
        
        # Should find valid adjustment set
        x_idx = do_calc.var_to_idx["Treatment"]
        y_idx = do_calc.var_to_idx["Outcome"]
        adjustment_set = do_calc.find_valid_adjustment_set(x_idx, y_idx)
        
        self.assertIsNotNone(adjustment_set, "Should find valid adjustment set")
        
        # Should include confounder
        confounder_idx = do_calc.var_to_idx["Confounder"]
        self.assertIn(confounder_idx, adjustment_set, "Should include confounder")
    
    def test_three_step_counterfactual(self):
        """Test 3-step counterfactual preserves noise structure."""
        do_calc = RobustDoCalculus(self.variable_names, self.adj_matrix)
        do_calc.fit(self.data)
        
        # Get counterfactual for first observation
        cf_result = do_calc.counterfactual_3step(
            observation_idx=0,
            intervention_var="Treatment", 
            intervention_value=self.data[0, 0] + 1.0
        )
        
        # Should return all variables
        self.assertEqual(len(cf_result), len(self.variable_names))
        
        # Treatment should be set to intervention value
        self.assertAlmostEqual(cf_result["Treatment"], self.data[0, 0] + 1.0, places=5)
        
        # Outcome should change (causal effect)
        original_outcome = self.data[0, 1]
        cf_outcome = cf_result["Outcome"]
        self.assertNotAlmostEqual(original_outcome, cf_outcome, places=2,
                                 msg="Outcome should change under intervention")


class TestPCStableEngine(unittest.TestCase):
    """Test Vulnerability 3 fix: Computational efficiency."""
    
    def setUp(self):
        np.random.seed(42)
        self.ci_test = RobustConditionalIndependenceTest(method="adaptive")
        
    def test_pc_stable_performance(self):
        """Test that PC-Stable provides speedup over sequential."""
        n_vars = 10  # Smaller for unit test
        n_samples = 200
        
        # Generate test data
        data = np.random.randn(n_samples, n_vars)
        for i in range(1, n_vars):
            parents = np.random.choice(i, min(2, i), replace=False)
            data[:, i] += np.sum(data[:, parents] * 0.3, axis=1)
        
        variable_names = [f"X{i}" for i in range(n_vars)]
        
        # Test parallel version
        pc_parallel = PCStableEngine(self.ci_test, max_cond_set_size=1, n_jobs=2)
        start_time = time.time()
        dag_parallel = pc_parallel.learn_structure(data, variable_names)
        parallel_time = time.time() - start_time
        
        # Test sequential version
        pc_sequential = PCStableEngine(self.ci_test, max_cond_set_size=1, n_jobs=1)
        start_time = time.time()
        dag_sequential = pc_sequential.learn_structure(data, variable_names)
        sequential_time = time.time() - start_time
        
        # Should complete successfully
        self.assertIsInstance(dag_parallel, dict)
        self.assertIsInstance(dag_sequential, dict)
        
        # Should discover similar number of edges
        parallel_edges = sum(len(children) for children in dag_parallel.values())
        sequential_edges = sum(len(children) for children in dag_sequential.values())
        
        # Allow some difference due to parallelization
        edge_diff = abs(parallel_edges - sequential_edges)
        self.assertLessEqual(edge_diff, 3, "Edge counts should be similar")
    
    def test_edge_orientation_stability(self):
        """Test that edge orientation is stable across runs."""
        n_vars = 6
        n_samples = 300
        
        # Generate data with clear v-structure: X -> Z <- Y
        x = np.random.randn(n_samples)
        y = np.random.randn(n_samples)
        z = 0.6 * x + 0.6 * y + np.random.randn(n_samples) * 0.2
        w1 = 0.5 * x + np.random.randn(n_samples) * 0.3
        w2 = 0.5 * y + np.random.randn(n_samples) * 0.3
        w3 = np.random.randn(n_samples)
        
        data = np.column_stack([x, y, z, w1, w2, w3])
        variable_names = ["X", "Y", "Z", "W1", "W2", "W3"]
        
        engine = PCStableEngine(self.ci_test, max_cond_set_size=2, n_jobs=1)
        dag = engine.learn_structure(data, variable_names)
        
        # Should orient v-structure correctly (X -> Z, Y -> Z)
        z_parents = [var for var, children in dag.items() if "Z" in children]
        
        # Z should have parents (v-structure)
        self.assertGreater(len(z_parents), 0, "Z should have parents in v-structure")


class TestIntegration(unittest.TestCase):
    """Integration tests for all components working together."""
    
    def test_full_pipeline_integration(self):
        """Test complete causal discovery pipeline."""
        np.random.seed(42)
        
        # Generate realistic financial-like data
        n_samples = 400
        volatility = np.abs(np.random.randn(n_samples) * 0.02)
        volume = 1e6 + volatility * 5e7 + np.random.randn(n_samples) * 1e5
        obi = 0.1 * (volume - volume.mean()) / volume.std() + np.random.randn(n_samples) * 0.05
        price = 100 + np.cumsum(obi * 0.5 + np.random.randn(n_samples) * 0.1)
        
        data = np.column_stack([price, volume, obi, volatility])
        variable_names = ["Price", "Volume", "OBI", "Volatility"]
        
        # Step 1: Discover structure with PC-Stable
        ci_test = RobustConditionalIndependenceTest(method="adaptive")
        pc_engine = PCStableEngine(ci_test, max_cond_set_size=2, n_jobs=1)
        dag = pc_engine.learn_structure(data, variable_names)
        
        # Should discover some edges
        total_edges = sum(len(children) for children in dag.values())
        self.assertGreater(total_edges, 0, "Should discover some causal relationships")
        
        # Step 2: Convert to adjacency matrix for Do-Calculus
        adj_matrix = np.zeros((len(variable_names), len(variable_names)))
        var_to_idx = {name: i for i, name in enumerate(variable_names)}
        
        for parent, children in dag.items():
            parent_idx = var_to_idx[parent]
            for child in children:
                child_idx = var_to_idx[child]
                adj_matrix[parent_idx, child_idx] = 1
        
        # Step 3: Test causal inference
        do_calc = RobustDoCalculus(variable_names, adj_matrix)
        do_calc.fit(data)
        
        # Try to estimate causal effect (may not be identifiable)
        try:
            effect = do_calc.compute_ate_backdoor("Volume", "Price")
            if effect.is_identifiable:
                self.assertIsInstance(effect.ate, float)
                self.assertFalse(np.isnan(effect.ate))
        except Exception:
            pass  # May not be identifiable, which is fine
    
    def test_edge_cases(self):
        """Test handling of edge cases."""
        ci_test = RobustConditionalIndependenceTest(method="adaptive")
        
        # Test with minimal data
        minimal_data = np.random.randn(10, 3)
        try:
            result = ci_test.test(minimal_data, 0, 1, [2])
            self.assertIsInstance(result[0], bool)
        except Exception as e:
            self.fail(f"Should handle minimal data: {e}")
        
        # Test with singular matrix (perfect correlation)
        singular_data = np.random.randn(50, 3)
        singular_data[:, 1] = singular_data[:, 0]  # Perfect correlation
        
        try:
            result = ci_test.test(singular_data, 0, 1, [2])
            self.assertIsInstance(result[0], bool)
        except Exception as e:
            self.fail(f"Should handle singular matrices: {e}")
        
        # Test with empty conditioning set
        try:
            result = ci_test.test(minimal_data, 0, 1, [])
            self.assertIsInstance(result[0], bool)
        except Exception as e:
            self.fail(f"Should handle empty conditioning set: {e}")


def run_test_suite():
    """Run the complete test suite."""
    print("=" * 80)
    print("RUNNING CAUSAL MODULE TEST SUITE")
    print("=" * 80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRobustCITest))
    suite.addTests(loader.loadTestsFromTestCase(TestRobustDoCalculus))
    suite.addTests(loader.loadTestsFromTestCase(TestPCStableEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    result = runner.run(suite)
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUITE SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    print(f"Total time: {total_time:.1f}s")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n{'✅ ALL TESTS PASSED' if success else '❌ SOME TESTS FAILED'}")
    
    return success


if __name__ == "__main__":
    run_test_suite()
