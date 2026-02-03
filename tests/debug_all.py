#!/usr/bin/env python3
"""
NeuralTrade - Comprehensive Debug & Validation Suite
======================================================
Tüm modüllerin çalışıp çalışmadığını test eder.
Direct imports - ana __init__.py zincirine bağımlı değil.
"""
import sys
import os

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import traceback
from datetime import datetime

try:
    from colorama import Fore, Style, init
    init()
except ImportError:
    # Colorama yoksa basit fallback
    class Fore:
        CYAN = GREEN = YELLOW = RED = ""
    class Style:
        RESET_ALL = ""

# Test sonuçları
results = {
    "passed": [],
    "failed": [],
    "skipped": []
}


def test_module(name: str, test_func):
    """Modül testi wrapper."""
    print(f"\n{Fore.CYAN}🧪 Testing: {name}{Style.RESET_ALL}", flush=True)
    try:
        test_func()
        results["passed"].append(name)
        print(f"{Fore.GREEN}  ✅ PASSED{Style.RESET_ALL}")
        return True
    except ImportError as e:
        results["skipped"].append((name, str(e)))
        print(f"{Fore.YELLOW}  ⚠️ SKIPPED: {e}{Style.RESET_ALL}")
        return False
    except Exception as e:
        results["failed"].append((name, str(e)))
        print(f"{Fore.RED}  ❌ FAILED: {e}{Style.RESET_ALL}")
        traceback.print_exc()
        return False


# ============================================================
# PHASE 19-23 TESTS
# ============================================================

def test_regime_hmm():
    """Phase 19: Regime Switching HMM"""
    # Direct import - not through modules package
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'modules'))
    from regime.hmm import HiddenMarkovModel, RegimeSwitcher
    
    hmm = HiddenMarkovModel(n_states=3)
    prices = [100 + i * 0.5 + (i % 10) for i in range(200)]
    
    # Calculate returns and volatilities
    returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
    volatilities = [abs(r) * 10 for r in returns]  # Simple volatility proxy
    
    hmm.fit(returns, volatilities)
    
    # Update with recent data
    hmm.update(returns[-1], volatilities[-1])
    result = hmm.get_current_regime()
    
    assert "name" in result
    assert "strategy" in result
    print(f"    Detected regime: {result['name']}")


def test_synthetic_gan():
    """Phase 20: Synthetic Data GAN"""
    from synthetic.generator import TimeSeriesGAN, StressTestEngine
    
    gan = TimeSeriesGAN()
    prices = [100 * (1 + 0.001 * i + 0.02 * (i % 5 - 2)) for i in range(200)]
    gan.fit(prices)
    
    synthetic = gan.generate(n_samples=2, length=50)
    assert len(synthetic) == 2
    assert len(synthetic[0]) == 50
    print(f"    Generated {len(synthetic)} synthetic series")


def test_tca():
    """Phase 21: Transaction Cost Analysis"""
    from tca.analyzer import TransactionCostAnalyzer
    
    tca = TransactionCostAnalyzer()
    result = tca.analyze_trade(
        decision_price=100,
        arrival_price=100.05,
        execution_price=100.10,
        post_trade_price=100.08,
        quantity=100,
        side="BUY",
        broker="TEST_BROKER"
    )
    
    assert "slippage_bps" in result
    print(f"    Slippage: {result['slippage_bps']:.2f} bps")


def test_distillation():
    """Phase 22: Knowledge Distillation"""
    from distillation.compressor import KnowledgeDistiller
    
    distiller = KnowledgeDistiller()
    distiller.teacher_outputs = [
        {"input": {"x": 1}, "output": {"logits": [0.1, 0.9]}},
        {"input": {"x": 2}, "output": {"logits": [0.8, 0.2]}},
    ]
    
    result = distiller.train_student(lambda: None, epochs=10)
    assert result.get("status") == "TRAINED"
    print(f"    Student trained: {result}")


def test_alpha_decay():
    """Phase 23: Alpha Decay Monitor"""
    from decay.monitor import AlphaDecayMonitor
    
    monitor = AlphaDecayMonitor()
    monitor.register_strategy("TEST_STRAT", "Test Strategy", initial_sharpe=2.0)
    
    for i in range(30):
        monitor.update_performance("TEST_STRAT", daily_return=0.001 - i * 0.0001)
    
    active = monitor.get_active_strategies()
    print(f"    Active strategies: {len(active)}")


# ============================================================
# PHASE 24-28 TESTS
# ============================================================

def test_walkforward():
    """Phase 24: Walk-Forward Optimization"""
    import numpy as np
    from validation.optimizer import WalkForwardOptimizer
    
    wfo = WalkForwardOptimizer(train_window=50, test_window=20, step_size=10)
    prices = np.array([100 * (1 + 0.001 * i) for i in range(150)])
    
    result = wfo.run(
        prices,
        optimize_func=lambda p: {"param": 1},
        backtest_func=lambda p, params: {"return": 0.05, "sharpe": 1.5}
    )
    
    assert "overfitting_ratio" in result
    print(f"    Overfit ratio: {result['overfitting_ratio']:.2f}")


def test_monte_carlo():
    """Phase 25: Monte Carlo Simulation"""
    from validation.optimizer import MonteCarloSimulator
    
    mc = MonteCarloSimulator(n_simulations=100)
    trade_returns = [0.01, -0.005, 0.02, -0.01, 0.015] * 20
    
    result = mc.simulate(trade_returns, initial_capital=10000)
    assert "ruin_probability" in result["risk_metrics"]
    print(f"    Ruin prob: {result['risk_metrics']['ruin_probability']:.2f}%")


def test_data_guard():
    """Phase 26-27: Data Integrity Guard"""
    from guard.integrity import DataIntegrityGuard
    
    guard = DataIntegrityGuard(max_price_change_pct=20)
    
    r1 = guard.validate_price("BTC", 50000)
    assert r1["is_valid"] == True
    
    r2 = guard.validate_price("BTC", 1)
    
    print(f"    Valid: {r1['is_valid']}, Spike rejected: {not r2['is_valid']}")


def test_rebalancer():
    """Phase 28: Dynamic Rebalancing"""
    from rebalance.dynamic import DynamicRebalancer
    
    rebalancer = DynamicRebalancer(threshold_pct=5)
    rebalancer.update_holdings({
        "STOCKS": 50000,
        "BONDS": 15000,
        "CRYPTO": 30000,
        "CASH": 5000
    })
    
    drift = rebalancer.calculate_drift()
    assert "needs_rebalance" in drift
    print(f"    Max drift: {drift['max_drift_pct']:.1f}%")


# ============================================================
# PHASE 29-33 TESTS
# ============================================================

def test_abm_simulation():
    """Phase 29: Agent-Based Market Simulation"""
    from simulation.abm import MarketSimulator
    
    sim = MarketSimulator(initial_price=100)
    sim.create_default_population(n_agents=100)
    
    result = sim.run_simulation(n_ticks=50)
    assert "final_price" in result
    print(f"    Final price: ${result['final_price']:.2f}")


def test_biometric():
    """Phase 30: Neuro-Finance Biometric"""
    from biometric.neurofinance import BiometricMonitor
    
    monitor = BiometricMonitor()
    metrics = monitor.simulate_wearable_data()
    result = monitor.update_metrics(metrics)
    
    assert "state" in result
    print(f"    Human state: {result['state']}")


def test_multicloud():
    """Phase 31-32: Multi-Cloud Geo-Arbitrage"""
    from cloud.multicloud import GeoArbitrageRouter, FailoverManager
    
    router = GeoArbitrageRouter()
    dc, latency = router.select_best_datacenter("BINANCE")
    
    failover = FailoverManager()
    status = failover.check_all_providers()
    
    print(f"    Best DC: {dc.name if dc else 'N/A'}, Latency: {latency:.1f}ms")


def test_vault():
    """Phase 33: Tokenized Vault"""
    from vault.tokenized import TokenizedVault
    
    vault = TokenizedVault("Test Vault")
    vault.deposit("0xABC123", 10000)
    vault.update_nav(11000)
    
    stats = vault.get_vault_stats()
    assert stats["roi_pct"] > 0
    print(f"    Vault ROI: {stats['roi_pct']:.1f}%")


# ============================================================
# PHASE 34-38 TESTS
# ============================================================

def test_tda():
    """Phase 34: Topological Data Analysis"""
    import numpy as np
    from topology.tda import PersistentHomology
    
    tda = PersistentHomology()
    prices = np.array([100 + i * 0.1 + np.sin(i / 5) * 2 for i in range(150)])
    
    result = tda.detect_topological_anomaly(prices)
    assert "beta_0" in result
    print(f"    Betti: β0={result['beta_0']}, β1={result['beta_1']}")


def test_causal():
    """Phase 35: Causal Inference"""
    from causal.inference import CausalGraph
    
    graph = CausalGraph()
    graph.build_financial_graph()
    
    result = graph.analyze_scenario("RATE_HIKE")
    assert "stock_impact_pct" in result
    print(f"    Rate hike impact: {result['stock_impact_pct']:.1f}%")


def test_gamma():
    """Phase 36: Gamma Exposure"""
    from options.gamma import GammaExposureEngine
    
    gex = GammaExposureEngine(spot_price=450)
    gex.add_option(440, 7, "PUT", 10000, 0.25)
    gex.add_option(460, 7, "CALL", 8000, 0.22)
    
    result = gex.calculate_total_gex()
    assert "total_gex_bn" in result
    print(f"    Total GEX: ${result['total_gex_bn']:.2f}B")


def test_ptp():
    """Phase 37: Precision Time Protocol"""
    import time as time_module
    from timing.ptp import StaleQuoteDetector
    
    detector = StaleQuoteDetector(stale_threshold_us=1000)
    detector.ptp.sync_with_grandmaster()
    
    result = detector.validate_quote(
        {"symbol": "BTC", "bid": 50000},
        time_module.time_ns() - 100000
    )
    
    print(f"    Quote status: {result['status']}")


def test_supply_chain():
    """Phase 38: Supply Chain GNN"""
    from graph.supply_chain import SupplyChainGraph
    
    graph = SupplyChainGraph()
    graph.build_tech_supply_chain()
    
    result = graph.analyze_event("taiwan_disaster")
    assert "trading_signals" in result
    print(f"    Cascade affected: {result['total_cascade_affected']} companies")


# ============================================================
# PHASE 39-43 TESTS
# ============================================================

def test_prediction_market():
    """Phase 39: Prediction Markets"""
    from prediction.markets import PredictionMarket
    
    market = PredictionMarket()
    market.register_agent("BULL_AGENT", 1000)
    market.register_agent("BEAR_AGENT", 1000)
    
    mkt_id = market.create_market("BTC up tomorrow?", ["YES", "NO"])
    market.place_bet(mkt_id, "BULL_AGENT", "YES", 100)
    market.place_bet(mkt_id, "BEAR_AGENT", "NO", 80)
    
    consensus = market.get_market_consensus(mkt_id)
    print(f"    Consensus: {consensus.get('consensus', 'N/A')}")


def test_formal_verification():
    """Phase 40: Formal Verification"""
    from formal.verification import FormalVerifier
    
    verifier = FormalVerifier()
    verifier.define_critical_invariants()
    
    result = verifier.verify_all_invariants()
    cb_proof = verifier.verify_circuit_breaker()
    
    print(f"    Invariants passed: {result['total_invariants'] - result['violations']}/{result['total_invariants']}")
    print(f"    CB theorem proven: {cb_proof['proven']}")


def test_quantum():
    """Phase 41: Quantum QPU"""
    import numpy as np
    from qpu.bridge import QuantumPortfolioOptimizer
    
    qpo = QuantumPortfolioOptimizer(backend="simulator")
    
    returns = np.array([0.10, 0.08, 0.12, 0.06])
    cov = np.array([
        [0.04, 0.01, 0.02, 0.01],
        [0.01, 0.03, 0.01, 0.01],
        [0.02, 0.01, 0.05, 0.02],
        [0.01, 0.01, 0.02, 0.02]
    ])
    
    result = qpo.optimize(returns, cov, risk_aversion=0.5)
    print(f"    Quantum weights: {[f'{w:.2f}' for w in result['weights']]}")


def test_sigint():
    """Phase 42: Corporate SIGINT"""
    from sigint.tracker import CorporateSIGINT
    
    sigint = CorporateSIGINT()
    
    sigint.adsb.track_flight("N889WM", "KLGB", "KOMA")
    sigint.adsb.track_flight("N1TM", "KSJC", "KOMA")
    
    convergences = sigint.adsb.detect_convergence()
    print(f"    Convergence events: {len(convergences)}")


def test_mesh():
    """Phase 43: Mesh Execution"""
    from mesh.executor import MeshExecutor
    
    mesh = MeshExecutor()
    status = mesh.check_all_channels()
    
    result = mesh.send_order({
        "symbol": "BTC",
        "side": "BUY",
        "quantity": 1,
        "price": 50000
    })
    
    print(f"    Order sent: {result['success']}, via: {result.get('channel', 'N/A')}")


# ============================================================
# ORCHESTRATOR TEST
# ============================================================

def test_orchestrator():
    """Orchestrator Test"""
    from orchestrator.adaptive import AdaptiveOrchestrator
    
    orch = AdaptiveOrchestrator()
    
    tier = orch.assess_complexity(
        {"volatility": 0.03, "regime": "NORMAL"},
        position_size_pct=5,
        urgency="NORMAL"
    )
    
    print(f"    Assessed tier: {tier.name}")


# ============================================================
# MAIN
# ============================================================

def main():
    """Ana test runner."""
    print(f"\n{'='*60}")
    print(f"🧪 NEURALTRADE DEBUG & VALIDATION SUITE")
    print(f"{'='*60}")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Project: {PROJECT_ROOT}")
    print(f"{'='*60}")
    
    # Add modules to path for direct imports
    modules_path = os.path.join(PROJECT_ROOT, 'modules')
    if modules_path not in sys.path:
        sys.path.insert(0, modules_path)
    
    # All tests
    tests = [
        # Phase 19-23
        ("Regime HMM", test_regime_hmm),
        ("Synthetic GAN", test_synthetic_gan),
        ("TCA", test_tca),
        ("Distillation", test_distillation),
        ("Alpha Decay", test_alpha_decay),
        
        # Phase 24-28
        ("Walk-Forward", test_walkforward),
        ("Monte Carlo", test_monte_carlo),
        ("Data Guard", test_data_guard),
        ("Rebalancer", test_rebalancer),
        
        # Phase 29-33
        ("ABM Simulation", test_abm_simulation),
        ("Biometric", test_biometric),
        ("Multi-Cloud", test_multicloud),
        ("Vault", test_vault),
        
        # Phase 34-38
        ("TDA", test_tda),
        ("Causal", test_causal),
        ("Gamma", test_gamma),
        ("PTP", test_ptp),
        ("Supply Chain", test_supply_chain),
        
        # Phase 39-43
        ("Prediction Market", test_prediction_market),
        ("Formal Verification", test_formal_verification),
        ("Quantum", test_quantum),
        ("SIGINT", test_sigint),
        ("Mesh", test_mesh),
        
        # Core
        ("Orchestrator", test_orchestrator),
    ]
    
    for name, test_func in tests:
        test_module(name, test_func)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"{Fore.GREEN}✅ PASSED: {len(results['passed'])}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}⚠️ SKIPPED: {len(results['skipped'])}{Style.RESET_ALL}")
    print(f"{Fore.RED}❌ FAILED: {len(results['failed'])}{Style.RESET_ALL}")
    
    if results["failed"]:
        print(f"\n{Fore.RED}FAILED TESTS:{Style.RESET_ALL}")
        for name, error in results["failed"]:
            print(f"  • {name}: {error}")
    
    total = len(tests)
    passed = len(results["passed"])
    
    print(f"\n{'='*60}")
    if passed == total:
        print(f"{Fore.GREEN}🎉 ALL TESTS PASSED! ({passed}/{total}){Style.RESET_ALL}")
    else:
        print(f"📈 Pass rate: {passed}/{total} ({passed/total*100:.0f}%)")
    print(f"{'='*60}\n")
    
    return len(results["failed"]) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
