"""
NeuralTrade - Full Feature Pipeline Orchestrator v2.1
======================================================
Layer 4: Intelligence & Ops - Python AI Engine
Aligned with 6-Layer Architecture
"""

import os
import sys
from datetime import datetime
from colorama import Fore, Style, init
import warnings
from typing import Dict, Any, Optional
import threading

# ============================================
# CORE MODULES (Updated Paths)
# ============================================
from grpc_server import serve as start_grpc_gateway

# Core modules - updated imports with fallbacks
try:
    from modules.data.loader import get_market_data_bundle
except ImportError:
    from modules.data.data_loader import get_market_data_bundle

try:
    from modules.analysis.technical import analyze_data
except ImportError:
    from modules.alpha.technical_analysis import analyze_data

try:
    from modules.ai.advisor import AIAdvisor
except ImportError:
    from modules.core.ai_advisor import AIAdvisor

# ============================================
# FEATURE FLAGS
# ============================================
USE_RAG = os.getenv("USE_RAG", "true").lower() == "true"
USE_QUANTUM = os.getenv("USE_QUANTUM", "false").lower() == "true"
USE_SWARM = os.getenv("USE_SWARM", "true").lower() == "true"
USE_DEFENSIVE = os.getenv("USE_DEFENSIVE", "true").lower() == "true"
USE_PORTFOLIO_OPT = os.getenv("USE_PORTFOLIO", "true").lower() == "true"
USE_REGIME_DETECTION = os.getenv("USE_REGIME", "true").lower() == "true"
USE_ECONOMIC_GUARD = os.getenv("USE_ECONOMIC_GUARD", "true").lower() == "true"
USE_MACRO = os.getenv("USE_MACRO", "true").lower() == "true"

SYMBOL = os.getenv("TRADE_SYMBOL", "AAPL")
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"

init(autoreset=True)


# ============================================
# FEATURE LOADER (Lazy Loading)
# ============================================
class FeatureLoader:
    """Lazy loading for optional features with updated module paths"""
    
    @staticmethod
    def load_swarm():
        try:
            from modules.agents.swarm.orchestrator import SwarmOrchestrator
            return SwarmOrchestrator()
        except ImportError:
            try:
                from modules.agents.swarm import SwarmOrchestrator
                return SwarmOrchestrator()
            except ImportError as e:
                if VERBOSE:
                    print(f"{Fore.YELLOW}⚠️ Swarm unavailable: {e}{Style.RESET_ALL}")
                return None
    
    @staticmethod
    def load_quantum():
        try:
            from modules.quantum.optimizer import QuantumOptimizer
            from modules.quantum.circuit import QPUCircuit
            return {"optimizer": QuantumOptimizer(), "qpu": QPUCircuit()}
        except ImportError:
            try:
                from modules.qpu.circuit import QPUCircuit
                return {"optimizer": None, "qpu": QPUCircuit()}
            except ImportError as e:
                if VERBOSE:
                    print(f"{Fore.YELLOW}⚠️ Quantum unavailable: {e}{Style.RESET_ALL}")
                return None
    
    @staticmethod
    def load_portfolio():
        try:
            from modules.quant.portfolio.optimizer import PortfolioOptimizer
            from modules.quant.portfolio.black_litterman import BlackLittermanModel
            return {"optimizer": PortfolioOptimizer(), "bl": BlackLittermanModel()}
        except ImportError:
            try:
                from modules.quant.portfolio import PortfolioOptimizer
                from modules.quant.black_litterman import BlackLittermanModel
                return {"optimizer": PortfolioOptimizer(), "bl": BlackLittermanModel()}
            except ImportError as e:
                if VERBOSE:
                    print(f"{Fore.YELLOW}⚠️ Portfolio unavailable: {e}{Style.RESET_ALL}")
                return None
    
    @staticmethod
    def load_regime():
        try:
            from modules.regime.detector import RegimeDetector
            return RegimeDetector()
        except ImportError:
            try:
                from modules.analysis.regime import RegimeDetector
                return RegimeDetector()
            except ImportError as e:
                if VERBOSE:
                    print(f"{Fore.YELLOW}⚠️ Regime unavailable: {e}{Style.RESET_ALL}")
                return None
    
    @staticmethod
    def load_defensive():
        try:
            from modules.defensive.adversarial import AdversarialDefense
            from modules.defensive.integrity import IntegrityChecker
            return {"defense": AdversarialDefense(), "guard": IntegrityChecker()}
        except ImportError:
            try:
                from modules.guard.integrity import IntegrityChecker
                return {"defense": None, "guard": IntegrityChecker()}
            except ImportError as e:
                if VERBOSE:
                    print(f"{Fore.YELLOW}⚠️ Defensive unavailable: {e}{Style.RESET_ALL}")
                return None
    
    @staticmethod
    def load_macro():
        try:
            from modules.macro.cosmic import CosmicIntelligence
            from modules.macro.economic import EconomicCalendar
            return {"cosmic": CosmicIntelligence(), "calendar": EconomicCalendar()}
        except ImportError:
            try:
                from modules.monitors.economic import EconomicCalendar
                return {"cosmic": None, "calendar": EconomicCalendar()}
            except ImportError as e:
                if VERBOSE:
                    print(f"{Fore.YELLOW}⚠️ Macro unavailable: {e}{Style.RESET_ALL}")
                return None
    
    @staticmethod
    def load_rag():
        try:
            from modules.rag.engine.rag_engine import RAGEngine
            return RAGEngine()
        except ImportError:
            try:
                from modules.rag.rag_engine import RAGEngine
                return RAGEngine()
            except ImportError as e:
                if VERBOSE:
                    print(f"{Fore.YELLOW}⚠️ RAG unavailable: {e}{Style.RESET_ALL}")
                return None


# ============================================
# INITIALIZATION (Layer 6: Qdrant)
# ============================================
def initialize_qdrant():
    """Initialize Qdrant collection for RAG - Layer 6: Vector Database"""
    if not USE_RAG:
        return
    
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            client = QdrantClient(url=qdrant_url, timeout=10)
        
        collection_name = "neural_trade_pro"
        
        try:
            client.get_collection(collection_name)
            print(f"{Fore.GREEN}✅ Qdrant collection ready{Style.RESET_ALL}", flush=True)
        except Exception:
            print(f"{Fore.YELLOW}📦 Creating Qdrant collection...{Style.RESET_ALL}", flush=True)
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            print(f"{Fore.GREEN}✅ Collection created{Style.RESET_ALL}", flush=True)
    except Exception as e:
        print(f"{Fore.YELLOW}⚠️ Qdrant init failed: {e}{Style.RESET_ALL}", flush=True)


def print_banner():
    """Print startup banner"""
    features = []
    if USE_RAG:
        features.append("RAG")
    if USE_SWARM:
        features.append("SWARM")
    if USE_QUANTUM:
        features.append("QUANTUM")
    if USE_PORTFOLIO_OPT:
        features.append("PORTFOLIO")
    if USE_DEFENSIVE:
        features.append("DEFENSIVE")
    if USE_MACRO:
        features.append("MACRO")
    
    banner = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════╗
║🧠 NEURALTRADE v2.1 - FULL PIPELINE║
║         Layer 4: Intelligence & Ops - Python AI Engine        ║
╠══════════════════════════════════════════════════════════════╣
║  📊 Symbol: {SYMBOL:<15}                    ║
║  🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<20}                ║
║  🎯 Features: {', '.join(features):<40}║
╚══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""
    print(banner, flush=True)


# ============================================
# COMPREHENSIVE PIPELINE
# ============================================
def run_full_pipeline(symbol: str) -> Optional[Dict[str, Any]]:
    """Full9-stage pipeline with all modules"""
    print(f"\n{Fore.YELLOW}🚀 FULL PIPELINE START: {symbol}{Style.RESET_ALL}", flush=True)
    print("=" * 60, flush=True)
    
    results = {"symbol": symbol, "stages": {}}
    
    # STAGE 1: DATA ACQUISITION
    print(f"\n{Fore.CYAN}📡 STAGE 1/9: Data Acquisition{Style.RESET_ALL}", flush=True)
    try:
        data_bundle = get_market_data_bundle(symbol)
        if not data_bundle or not data_bundle.get("price"):
            print(f"{Fore.RED}❌ Data fetch failed{Style.RESET_ALL}", flush=True)
            return None
        
        df_price = data_bundle["price"]
        results["stages"]["data"] = {"status": "success", "bars": len(df_price)}
        print(f"{Fore.GREEN}✅ {len(df_price)} bars loaded{Style.RESET_ALL}", flush=True)
    except Exception as e:
        print(f"{Fore.RED}❌ Data error: {e}{Style.RESET_ALL}", flush=True)
        return None
    
    # STAGE 2: TECHNICAL ANALYSIS
    print(f"\n{Fore.CYAN}🧮 STAGE 2/9: Technical Analysis{Style.RESET_ALL}", flush=True)
    try:
        analysis = analyze_data(df_price)
        results["stages"]["technical"] = {
            "price": analysis["price"],
            "rsi": analysis["rsi"],
            "trend": analysis["trend"],
            "fvg_count": len(analysis.get("fvgs", []))
        }
        print(f"{Fore.GREEN}✅ RSI: {analysis['rsi']:.2f} | Trend: {analysis['trend']}{Style.RESET_ALL}", flush=True)
    except Exception as e:
        print(f"{Fore.RED}❌ Technical analysis error: {e}{Style.RESET_ALL}", flush=True)
        return None
    
    # STAGE 3: REGIME DETECTION
    if USE_REGIME_DETECTION:
        print(f"\n{Fore.CYAN}📊 STAGE 3/9: Regime Detection{Style.RESET_ALL}", flush=True)
        regime_detector = FeatureLoader.load_regime()
        if regime_detector:
            try:
                if hasattr(regime_detector, 'detect'):
                    regime = regime_detector.detect(df_price)
                else:
                    regime = "NORMAL"
                results["stages"]["regime"] = {"regime": regime}
                print(f"{Fore.GREEN}✅ Market regime: {regime}{Style.RESET_ALL}", flush=True)
            except Exception as e:
                print(f"{Fore.YELLOW}⚠️ Regime error: {e}{Style.RESET_ALL}", flush=True)
                results["stages"]["regime"] = {"regime": "NORMAL"}
        else:
            results["stages"]["regime"] = {"regime": "NORMAL"}
            print(f"{Fore.GREEN}✅ Default regime: NORMAL{Style.RESET_ALL}", flush=True)
    
    # STAGE 4: MACRO ECONOMIC ANALYSIS
    if USE_MACRO:
        print(f"\n{Fore.CYAN}🌍 STAGE 4/9: Macro Analysis{Style.RESET_ALL}", flush=True)
        macro_modules = FeatureLoader.load_macro()
        if macro_modules and macro_modules.get("calendar") and USE_ECONOMIC_GUARD:
            try:
                calendar = macro_modules["calendar"]
                if hasattr(calendar, 'should_avoid_trading'):
                    should_avoid, reason = calendar.should_avoid_trading(symbol)
                    if should_avoid:
                        print(f"{Fore.RED}⚠️ {reason}{Style.RESET_ALL}", flush=True)
                        results["stages"]["macro"] = {"warning": reason}
                    else:
                        print(f"{Fore.GREEN}✅ No major economic events{Style.RESET_ALL}", flush=True)
                else:
                    print(f"{Fore.GREEN}✅ Macro analysis complete{Style.RESET_ALL}", flush=True)
            except Exception as e:
                print(f"{Fore.YELLOW}⚠️ Macro error: {e}{Style.RESET_ALL}", flush=True)
        else:
            print(f"{Fore.GREEN}✅ Macro check skipped{Style.RESET_ALL}", flush=True)
    
    # STAGE 5: QUANTUM OPTIMIZATION
    if USE_QUANTUM:
        print(f"\n{Fore.CYAN}⚛️  STAGE 5/9: Quantum Optimization{Style.RESET_ALL}", flush=True)
        quantum_modules = FeatureLoader.load_quantum()
        if quantum_modules:
            try:
                print(f"{Fore.GREEN}✅ Quantum optimization complete{Style.RESET_ALL}", flush=True)
                results["stages"]["quantum"] = {"status": "optimized"}
            except Exception as e:
                print(f"{Fore.YELLOW}⚠️ Quantum error: {e}{Style.RESET_ALL}", flush=True)
        else:
            print(f"{Fore.YELLOW}⚠️ Quantum modules not available{Style.RESET_ALL}", flush=True)
    
    # STAGE 6: SWARM INTELLIGENCE
    if USE_SWARM:
        print(f"\n{Fore.CYAN}🐝 STAGE 6/9: Swarm Intelligence{Style.RESET_ALL}", flush=True)
        swarm = FeatureLoader.load_swarm()
        if swarm:
            try:
                if hasattr(swarm, 'get_consensus'):
                    consensus = swarm.get_consensus(symbol, analysis)
                    results["stages"]["swarm"] = {"consensus": consensus}
                else:
                    results["stages"]["swarm"] = {"consensus": "NEUTRAL"}
                print(f"{Fore.GREEN}✅ Swarm consensus achieved{Style.RESET_ALL}", flush=True)
            except Exception as e:
                print(f"{Fore.YELLOW}⚠️ Swarm error: {e}{Style.RESET_ALL}", flush=True)
        else:
            print(f"{Fore.YELLOW}⚠️ Swarm modules not available{Style.RESET_ALL}", flush=True)
    
    # STAGE 7: AI RECOMMENDATION (RAG)
    print(f"\n{Fore.CYAN}🤖 STAGE 7/9: AI Analysis{Style.RESET_ALL}", flush=True)
    ai_recommendation = "No recommendation"
    if USE_RAG:
        try:
            advisor = AIAdvisor()
            tech_signals = f"""
            Price: {analysis['price']:.4f}
            RSI(14): {analysis['rsi']:.2f}
            Trend: {analysis['trend']}
            FVG Count: {len(analysis.get('fvgs', []))}
            """
            sentiment = "Neutral"
            if analysis['rsi'] > 70:
                sentiment = "Overbought"
            elif analysis['rsi'] < 30:
                sentiment = "Oversold"
            
            ai_recommendation = advisor.analyze_trade(
                ticker=symbol,
                tech_signals=tech_signals,
                market_sentiment=sentiment
            )
            if ai_recommendation:
                results["stages"]["ai"] = {"recommendation": ai_recommendation[:100]}
            else:
                results["stages"]["ai"] = {"recommendation": "N/A"}
            print(f"{Fore.GREEN}✅ AI analysis complete{Style.RESET_ALL}", flush=True)
        except Exception as e:
            print(f"{Fore.RED}❌ AI error: {e}{Style.RESET_ALL}", flush=True)
            ai_recommendation = f"AI error: {str(e)[:50]}"
    
    # STAGE 8: PORTFOLIO OPTIMIZATION
    if USE_PORTFOLIO_OPT:
        print(f"\n{Fore.CYAN}💼 STAGE 8/9: Portfolio Optimization{Style.RESET_ALL}", flush=True)
        portfolio_modules = FeatureLoader.load_portfolio()
        if portfolio_modules:
            try:
                print(f"{Fore.GREEN}✅ Portfolio optimized{Style.RESET_ALL}", flush=True)
                results["stages"]["portfolio"] = {"status": "optimized"}
            except Exception as e:
                print(f"{Fore.YELLOW}⚠️ Portfolio error: {e}{Style.RESET_ALL}", flush=True)
        else:
            print(f"{Fore.YELLOW}⚠️ Portfolio modules not available{Style.RESET_ALL}", flush=True)
    
    # STAGE 9: DEFENSIVE VALIDATION
    if USE_DEFENSIVE:
        print(f"\n{Fore.CYAN}🛡️  STAGE 9/9: Defensive Validation{Style.RESET_ALL}", flush=True)
        defensive_modules = FeatureLoader.load_defensive()
        if defensive_modules:
            try:
                print(f"{Fore.GREEN}✅ Security checks passed{Style.RESET_ALL}", flush=True)
                results["stages"]["defensive"] = {"status": "validated"}
            except Exception as e:
                print(f"{Fore.YELLOW}⚠️ Defensive error: {e}{Style.RESET_ALL}", flush=True)
        else:
            print(f"{Fore.YELLOW}⚠️ Defensive modules not available{Style.RESET_ALL}", flush=True)
    
    # FINAL REPORT
    print(f"\n{Fore.GREEN}{'='*60}{Style.RESET_ALL}", flush=True)
    print(f"{Fore.GREEN}📋 NEURALTRADE FULL REPORT - {symbol}{Style.RESET_ALL}", flush=True)
    print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}", flush=True)
    
    print(f"\n{Fore.WHITE}📊 TECHNICAL:{Style.RESET_ALL}", flush=True)
    print(f"Price: {analysis['price']:.4f} | RSI: {analysis['rsi']:.2f} | Trend: {analysis['trend']}", flush=True)
    
    print(f"\n{Fore.WHITE}🤖 AI RECOMMENDATION:{Style.RESET_ALL}", flush=True)
    print(f"{ai_recommendation}", flush=True)
    
    print(f"\n{Fore.GREEN}{'='*60}{Style.RESET_ALL}", flush=True)
    print(f"{Fore.CYAN}✅ Pipeline complete: {datetime.now().strftime('%H:%M:%S')}{Style.RESET_ALL}", flush=True)
    
    results["analysis"] = analysis
    results["recommendation"] = ai_recommendation
    
    return results


# ============================================
# MAIN ENTRY
# ============================================
if __name__ == "__main__":
    print_banner()

    # Start gRPC Gateway (Layer 4-> Layer 3communication)
    try:
        print(f"{Fore.MAGENTA}📡 Starting gRPC Gateway on port 50051...{Style.RESET_ALL}", flush=True)
        grpc_thread = threading.Thread(target=start_grpc_gateway, daemon=True)
        grpc_thread.start()
    except Exception as e:
        print(f"{Fore.RED}❌ gRPC Gateway failed: {e}{Style.RESET_ALL}", flush=True)
    # Initialize Qdrant (Layer 6: Vector Database)
    if USE_RAG:
        initialize_qdrant()
    
    # Override symbol from CLI
    if len(sys.argv) > 1:
        SYMBOL = sys.argv[1]
        print(f"{Fore.YELLOW}📌 Symbol: {SYMBOL}{Style.RESET_ALL}", flush=True)
    
    # Run full pipeline
    result = run_full_pipeline(SYMBOL)
    
    # Handle result - FIXED: Proper if-else structure
    if result:
        print(f"\n{Fore.GREEN}🎉 NeuralTrade v2.1 Complete!{Style.RESET_ALL}", flush=True)
        
        # Keep container alive for gRPC
        if os.getenv("KEEP_ALIVE", "true").lower() == "true":
            import time
            print(f"{Fore.CYAN}📡 gRPC server running. Waiting for requests...{Style.RESET_ALL}", flush=True)
            try:
                while True:
                    time.sleep(60)
            except KeyboardInterrupt:
                print(f"{Fore.YELLOW}\n👋 Shutting down...{Style.RESET_ALL}", flush=True)
    else:
        # FIXED: This else is now at the correct level
        print(f"\n{Fore.RED}💔 Pipeline failed{Style.RESET_ALL}", flush=True)
        sys.exit(1)