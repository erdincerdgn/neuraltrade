"""
NeuralTrade FastAPI Server
==========================
REST API and gRPC server for Python AI Engine.
Wraps the pipeline orchestrator for NestJS integration.

Port 8000: REST API
Port 50051: gRPC Server
"""

import os
import sys


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # GPU arama, CPU kullan
# os.environ["USE_TF"] = "0"                 # Transformers'a TF'yi yasakla
# os.environ["USE_TORCH"] = "1"              # Sadece Torch kullan
# os.environ["TRANSFORMERS_NO_TF"] = "1"     # Ekstra güvenlik
# os.environ["HF_HOME"] = "/tmp/huggingface" # Cache izni

try:
    import torch
    # Eğer CUDA versiyonu None ise (CPU sürümü yüklüyse), sahte bir versiyon atıyoruz.
    if getattr(torch.version, "cuda", None) is None:
        torch.version.cuda = "11.8"
    
    # Bazen __version__ da eksik olabiliyor, onu da garantiye alalım
    if not hasattr(torch, "__version__"):
        torch.__version__ = "2.1.0"
except ImportError:
    pass # Torch yüklü değilse zaten birazdan hata verir, burayı geç.





import random
import asyncio
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from prometheus_fastapi_instrumentator import Instrumentator

from infrastructure.prometheus.metrics.metrics import (
    portfolio_gauge,
    active_trades_gauge,
    daily_pnl_gauge,
    win_rate_gauge,
    ai_confidence_gauge,
    latency_gauge
)

# ============================================
# SIMULATION ENGINE 🏎️
# ============================================
async def simulate_live_market():
    """
    Simulates live market data to populate Grafana dashboard
    with dynamic, realistic-looking values.
    """
    current_balance = 100000.00 # Start with 100k
    current_pnl = 0.0
    
    print("📈 Market Simulation Started (English Mode)...")
    
    while True:
        # 1. Portfolio Movement (High Volatility)
        change = random.uniform(-2000, 5000) 
        current_balance += change
        current_pnl += change 
        
        # 2. Generate Random Metrics
        win_rate = random.uniform(55.0, 78.0)       # Success between 55-78%
        ai_confidence = random.uniform(80.0, 99.9)  # High confidence
        latency = random.uniform(15, 120)           # Speed in ms
        
        # 3. Push to Prometheus (Artık metrics.py'dan gelen objeleri kullanıyor)
        portfolio_gauge.set(round(current_balance, 2))
        daily_pnl_gauge.set(round(current_pnl, 2))
        win_rate_gauge.set(round(win_rate, 1))
        ai_confidence_gauge.set(round(ai_confidence, 1))
        latency_gauge.set(int(latency))
        active_trades_gauge.set(random.randint(3, 12))
        
        # Update every 2 seconds
        await asyncio.sleep(2)


# ============================================
# Pydantic Models
# ============================================

class SignalRequest(BaseModel):
    symbol: str
    timeframe: str = "1d"
    use_quantum: bool = False
    use_swarm: bool = True
    use_regime: bool = True
    use_macro: bool = True


class SignalResponse(BaseModel):
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    models: List[str]
    reasoning: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None


class MarketAnalysisRequest(BaseModel):
    symbol: str
    query: Optional[str] = None


class MarketAnalysisResponse(BaseModel):
    symbol: str
    summary: str
    sentiment: str  # bullish, bearish, neutral
    key_factors: List[str]
    sources: List[str]


class PortfolioHolding(BaseModel):
    symbol: str
    quantity: float
    current_price: Optional[float] = None
    entry_price: Optional[float] = None


class PortfolioOptimizationRequest(BaseModel):
    holdings: List[PortfolioHolding]


class RiskAssessmentResponse(BaseModel):
    overall_risk: str
    var_95: float
    max_drawdown: float
    beta: float
    diversification_score: float
    warnings: List[str]


class PipelineRequest(BaseModel):
    symbol: str


class HealthResponse(BaseModel):
    status: str
    version: str
    features: Dict[str, bool]


# ============================================
# Import AI Modules (Lazy)
# ============================================

def get_pipeline():
    """Lazy load the pipeline to avoid circular imports"""
    try:
        from modules.data.data_loader import get_market_data_bundle
        from modules.alpha.technical_analysis import analyze_data
        from modules.core.ai_advisor import AIAdvisor
        return {
            "data_loader": get_market_data_bundle,
            "analyzer": analyze_data,
            "advisor": AIAdvisor()
        }
    except ImportError as e:
        print(f"Warning: Could not import AI modules: {e}")
        return None


def get_swarm():
    """Load swarm intelligence"""
    try:
        from modules.agents.swarm import SwarmOrchestrator
        return SwarmOrchestrator()
    except ImportError:
        return None


def get_rag():
    """Load RAG module"""
    try:
        from modules.rag.graph import RAGPipeline
        return RAGPipeline()
    except ImportError:
        return None


# ============================================
# FastAPI Application
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    print("🚀 NeuralTrade AI Server starting...")

    # 1. Prometheus Metriklerini Başlat (BURAYA TAŞIDIK) ✅
    

    # 2. Simülasyon Motorunu Ateşle (BURAYA TAŞIDIK) ✅
    asyncio.create_task(simulate_live_market())
    
    # Initialize components
    app.state.pipeline = get_pipeline()
    app.state.swarm = get_swarm()
    app.state.rag = get_rag()
    
    # Initialize Qdrant if RAG is enabled
    if os.getenv("USE_RAG", "true").lower() == "true":
        try:
            from main import initialize_qdrant
            initialize_qdrant()
        except Exception as e:
            print(f"Warning: Qdrant init failed: {e}")
    
    print("✅ NeuralTrade AI Server ready")
    yield
    print("👋 NeuralTrade AI Server shutting down...")


app = FastAPI(
    title="NeuralTrade AI Engine",
    description="Python AI Engine for NeuralTrade Trading Platform",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

# ============================================
# Health & Metrics Endpoints
# ============================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for Docker/K8s"""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        features={
            "rag": os.getenv("USE_RAG", "true").lower() == "true",
            "quantum": os.getenv("USE_QUANTUM", "false").lower() == "true",
            "swarm": os.getenv("USE_SWARM", "true").lower() == "true",
            "defensive": os.getenv("USE_DEFENSIVE", "true").lower() == "true",
            "macro": os.getenv("USE_MACRO", "true").lower() == "true",
        }
    )


@app.get("/metrics-custom")
async def metrics_custom():
    """Prometheus metrics endpoint"""
    # TODO: Add prom-client integration
    return {
        "requests_total": 0,
        "signals_generated": 0,
        "pipeline_runs": 0,
    }


# ============================================
# Signal Generation Endpoints
# ============================================

@app.post("/signal/generate", response_model=SignalResponse)
async def generate_signal(request: SignalRequest):
    """Generate trading signal for a symbol"""
    from datetime import datetime
    
    pipeline = app.state.pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="AI pipeline not available")
    
    try:
        # Get market data
        data = pipeline["data_loader"](request.symbol)
        if data is None:
            raise HTTPException(status_code=404, detail=f"No data for {request.symbol}")
        
        # Technical analysis
        analysis = pipeline["analyzer"](data)
        
        # AI advisor signal
        advisor = pipeline["advisor"]
        signal = advisor.get_signal(request.symbol, data, analysis)
        
        # Swarm consensus (if enabled)
        swarm_consensus = None
        if request.use_swarm and app.state.swarm:
            try:
                swarm_result = app.state.swarm.get_consensus(request.symbol, data)
                swarm_consensus = swarm_result.get("consensus", 0.5)
            except Exception:
                pass
        
        return SignalResponse(
            symbol=request.symbol,
            action=signal.get("action", "HOLD"),
            confidence=signal.get("confidence", 0.5),
            models=signal.get("models", ["ensemble"]),
            reasoning=signal.get("reasoning", "Based on technical and AI analysis"),
            timestamp=datetime.utcnow().isoformat(),
            metadata={
                "swarm_consensus": swarm_consensus,
                "regime": signal.get("regime"),
                "risk_level": signal.get("risk_level"),
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/signal/batch")
async def batch_signals(symbols: List[str]):
    """Generate signals for multiple symbols"""
    results = []
    for symbol in symbols:
        try:
            signal = await generate_signal(SignalRequest(symbol=symbol))
            results.append(signal)
        except HTTPException as e:
            results.append({"symbol": symbol, "error": e.detail})
    return results


# ============================================
# Market Analysis Endpoints
# ============================================

@app.post("/market/analyze", response_model=MarketAnalysisResponse)
async def analyze_market(request: MarketAnalysisRequest):
    """Analyze market for a symbol using RAG"""
    rag = app.state.rag
    
    if rag is None:
        # Fallback without RAG
        return MarketAnalysisResponse(
            symbol=request.symbol,
            summary=f"Basic analysis for {request.symbol}",
            sentiment="neutral",
            key_factors=["Price action", "Volume", "Technical indicators"],
            sources=["Internal analysis"],
        )
    
    try:
        query = request.query or f"What is the market outlook for {request.symbol}?"
        result = rag.query(query, context={"symbol": request.symbol})
        
        return MarketAnalysisResponse(
            symbol=request.symbol,
            summary=result.get("answer", ""),
            sentiment=result.get("sentiment", "neutral"),
            key_factors=result.get("factors", []),
            sources=result.get("sources", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/technical/{symbol}")
async def technical_analysis(symbol: str):
    """Get technical analysis for a symbol"""
    pipeline = app.state.pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="AI pipeline not available")
    
    try:
        data = pipeline["data_loader"](symbol)
        if data is None:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")
        
        analysis = pipeline["analyzer"](data)
        return {
            "symbol": symbol,
            "trend": analysis.get("trend", "sideways"),
            "support": analysis.get("support", []),
            "resistance": analysis.get("resistance", []),
            "indicators": analysis.get("indicators", {}),
            "patterns": analysis.get("patterns", []),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Portfolio Endpoints
# ============================================

@app.post("/portfolio/optimize")
async def optimize_portfolio(request: PortfolioOptimizationRequest):
    """Optimize portfolio allocation"""
    try:
        from modules.quant.portfolio import PortfolioOptimizer
        optimizer = PortfolioOptimizer()
        
        holdings_dict = {h.symbol: h.quantity for h in request.holdings}
        result = optimizer.optimize(holdings_dict)
        
        return {
            "current_allocation": holdings_dict,
            "optimal_allocation": result.get("optimal", {}),
            "expected_return": result.get("expected_return", 0),
            "expected_risk": result.get("expected_risk", 0),
            "sharpe_ratio": result.get("sharpe_ratio", 0),
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="Portfolio optimizer not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/portfolio/risk", response_model=RiskAssessmentResponse)
async def assess_risk(request: PortfolioOptimizationRequest):
    """Assess portfolio risk"""
    try:
        from modules.quant.risk import RiskAnalyzer
        analyzer = RiskAnalyzer()
        
        holdings_dict = {h.symbol: h.quantity for h in request.holdings}
        result = analyzer.analyze(holdings_dict)
        
        return RiskAssessmentResponse(
            overall_risk=result.get("risk_level", "moderate"),
            var_95=result.get("var_95", 0.0),
            max_drawdown=result.get("max_drawdown", 0.0),
            beta=result.get("beta", 1.0),
            diversification_score=result.get("diversification", 0.5),
            warnings=result.get("warnings", []),
        )
    except ImportError:
        return RiskAssessmentResponse(
            overall_risk="unknown",
            var_95=0.0,
            max_drawdown=0.0,
            beta=1.0,
            diversification_score=0.0,
            warnings=["Risk analyzer not available"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Pipeline Endpoint
# ============================================

@app.post("/pipeline/run")
async def run_pipeline(request: PipelineRequest, background_tasks: BackgroundTasks):
    """Run the full AI pipeline for a symbol"""
    try:
        from main import run_full_pipeline
        
        # Run in background for long-running pipelines
        background_tasks.add_task(run_full_pipeline, request.symbol)
        
        return {
            "status": "started",
            "symbol": request.symbol,
            "message": f"Pipeline started for {request.symbol}",
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="Pipeline not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# RAG Query Endpoint
# ============================================

@app.post("/rag/query")
async def query_rag(query: str, context: Optional[str] = None):
    """Query the RAG system"""
    rag = app.state.rag
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG not available")
    
    try:
        result = rag.query(query, context=context)
        return {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "confidence": result.get("confidence", 0.0),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Main Entry Point
# ============================================

if __name__ == "__main__":

    import threading
    from grpc_server import serve as start_grpc_gateway

    grpc_thread = threading.Thread(target=start_grpc_gateway, daemon=True)
    grpc_thread.start()
    print(f"📡 gRPC Gateway started on port 50051")

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    print(f"🚀 Starting NeuralTrade AI Server on {host}:{port}")
    
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=os.getenv("API_RELOAD", "false").lower() == "true",
        workers=int(os.getenv("API_WORKERS", "1")),
    )
