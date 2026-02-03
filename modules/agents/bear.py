"""
Bear Agent - Bearish Market Sentiment Analyzer
Author: Erdinc Erdogan
Purpose: Identifies risks, downside scenarios, and bearish signals in financial markets by analyzing
technical indicators, macro risks, and sentiment data to advocate cautious trading positions.
References:
- Technical Analysis Indicators (RSI, Moving Averages, Volume Analysis)
- Behavioral Finance and Market Sentiment Theory
- Risk Assessment and Position Management
Usage:
    bear = BearAgent(llm=ollama_llm)
    result = bear.analyze(ticker="AAPL", price_data=data, technicals=tech_signals)
"""
import os
from typing import Dict, List, Optional
from datetime import datetime
from colorama import Fore, Style


class BearAgent:
    """
    Bear Agent.
    
    Task: Identify risks and downside scenarios in the market.
    Always advocates caution and highlights potential dangers.
    """
    
    # Bearish indicators
    BEARISH_INDICATORS = {
        "rsi_overbought": {"condition": "rsi > 70", "weight": 0.8, "reason": "RSI is in the overbought zone - a decline is expected"},
        "death_cross": {"condition": "ma50 < ma200", "weight": 0.9, "reason": "Death cross - strong bearish signal"},
        "volume_decline": {"condition": "volume < avg_volume * 0.5", "weight": 0.5, "reason": "Low volume - waning interest"},
        "resistance_rejection": {"condition": "price rejected at resistance", "weight": 0.75, "reason": "Rejection at resistance level"},
        "bearish_divergence": {"condition": "price up, rsi down", "weight": 0.85, "reason": "Bearish divergence"},
        "earnings_miss": {"condition": "eps < expected", "weight": 0.8, "reason": "Earnings below expectations"},
        "insider_selling": {"condition": "net_insider < 0", "weight": 0.7, "reason": "Insider selling"},
        "sector_outflow": {"condition": "sector_outflow", "weight": 0.6, "reason": "Capital outflow from sector"},
    }
    
    # Macro risk factors
    MACRO_RISKS = [
        "fed_hawkish",        # FED hawkish
        "inflation_rising",   # Inflation rising
        "recession_fear",     # Recession fear
        "geopolitical_risk",  # Geopolitical risks
        "debt_crisis",        # Debt crisis
        "bank_failure",       # Bank failure
    ]
    
    def __init__(self, llm=None):
        """
        Args:
            llm: LangChain LLM instance
        """
        self.llm = llm
        self.name = "🐻 BEAR AGENT"
        self.bias = "BEARISH"
        self.confidence = 0.0
        self.arguments = []
    
    def analyze(self,
                ticker: str,
                price_data: Dict,
                news: List[str] = None,
                technicals: Dict = None) -> Dict:
        """
        Analyze from a risk perspective.
        
        Args:
            ticker: Stock symbol
            price_data: Price data
            news: Recent news
            technicals: Technical indicators
        """
        print(f"{Fore.RED}{self.name}: Analyzing risk for {ticker}...{Style.RESET_ALL}", flush=True)
        
        self.arguments = []
        total_weight = 0
        
        tech = technicals or {}
        
        # Check technical indicators
        # RSI
        rsi = tech.get("rsi", 50)
        if rsi > 70:
            self._add_argument("rsi_overbought", rsi)
            total_weight += 0.8
        elif rsi > 60:
            self._add_argument("rsi_high", rsi, "RSI is high - risk of overbought")
            total_weight += 0.4
        
        # Trend
        trend = tech.get("trend", "").lower()
        if "bear" in trend or "dusus" in trend.lower() if trend else False:
            self._add_argument("trend_bearish", trend, "Bearish trend active")
            total_weight += 0.6
        
        # Price momentum negative
        price = price_data.get("close", price_data.get("price", 0))
        prev_price = price_data.get("prev_close", price * 1.01)
        if price < prev_price:
            change_pct = ((price - prev_price) / prev_price) * 100
            self._add_argument("negative_momentum", change_pct, f"Negative momentum: {change_pct:.2f}%")
            total_weight += 0.4
        
        # High volatility
        volatility = tech.get("volatility", 0)
        if volatility > 0.03:  # More than 3%
            self._add_argument("high_volatility", volatility*100, f"High volatility: %{volatility*100:.1f}")
            total_weight += 0.5
        
        # FVG analysis - Bearish
        fvgs = tech.get("fvgs", [])
        bearish_fvgs = [f for f in fvgs if f.get("type", "").lower() == "bearish"]
        if bearish_fvgs:
            self._add_argument("bearish_fvg", len(bearish_fvgs), f"{len(bearish_fvgs)} bearish FVG found")
            total_weight += 0.4
        
        # Drawdown risk
        high_52w = price_data.get("high_52w", price)
        if high_52w > 0:
            drawdown = ((high_52w - price) / high_52w) * 100
            if drawdown > 20:
                self._add_argument("significant_drawdown", drawdown, f"Drawdown from 52-week high: %{drawdown:.0f}")
                total_weight += 0.6
        
        # News risk analysis
        if news:
            bearish_news = self._analyze_news_risks(news)
            if bearish_news:
                self._add_argument("negative_news", bearish_news, f"Negative news: {len(bearish_news)}")
                total_weight += 0.4 * len(bearish_news)
        
        # LLM risk analysis
        if self.llm:
            llm_analysis = self._llm_bearish_analysis(ticker, price_data, technicals)
            if llm_analysis:
                self.arguments.append(llm_analysis)
                total_weight += 0.5
        
        # Calculate confidence
        max_weight = len(self.BEARISH_INDICATORS) * 0.8
        self.confidence = min(total_weight / max_weight, 1.0) if max_weight > 0 else 0.5
        
        # Result
        result = {
            "agent": self.name,
            "ticker": ticker,
            "recommendation": "SELL" if self.confidence > 0.6 else "HOLD" if self.confidence > 0.4 else "BUY",
            "confidence": self.confidence,
            "bias": self.bias,
            "arguments": self.arguments,
            "risk_level": self._calculate_risk_level(),
            "summary": self._generate_summary(ticker)
        }
        
        print(f"{Fore.RED}{self.name}: {result['recommendation']} (Risk: {result['risk_level']}){Style.RESET_ALL}", flush=True)
        
        return result
    
    def _add_argument(self, indicator: str, value, reason: str = None):
        """Add an argument."""
        if reason is None:
            reason = self.BEARISH_INDICATORS.get(indicator, {}).get("reason", indicator)
        
        self.arguments.append({
            "indicator": indicator,
            "value": value,
            "reason": reason,
            "sentiment": "BEARISH"
        })
    
    def _analyze_news_risks(self, news: List[str]) -> List[str]:
        """Search for risk signals in news."""
        risk_keywords = [
            "crash", "fall", "drop", "concern", "warning", "risk",
            "downgrade", "sell", "investigation", "lawsuit", "fraud",
            "layoff", "bankruptcy", "default", "miss", "decline"
        ]
        
        risky_news = []
        for article in news:
            article_lower = article.lower()
            if any(kw in article_lower for kw in risk_keywords):
                risky_news.append(article)
        
        return risky_news[:3]
    
    def _llm_bearish_analysis(self, ticker: str, price_data: Dict, technicals: Dict) -> Optional[Dict]:
        """LLM risk analysis."""
        if not self.llm:
            return None
        
        prompt = f"""You are a BEAR analyst and a RISK HUNTER. Analyze ONLY the risks and downside scenarios for the {ticker} stock.

Technical Data:
- Price: ${price_data.get('price', price_data.get('close', 0)):.2f}
- RSI: {technicals.get('rsi', 'N/A')}
- Trend: {technicals.get('trend', 'N/A')}

Task: Identify the top 2-3 critical risk factors. Focus only on the negative perspective.
Format: Brief and concise, bullet points."""

        try:
            response = self.llm.invoke(prompt)
            return {
                "indicator": "llm_risk_analysis",
                "value": response[:500],
                "reason": "AI In-Depth Risk Analysis",
                "sentiment": "BEARISH"
            }
        except:
            return None
    
    def _calculate_risk_level(self) -> str:
        """Calculate risk level."""
        if self.confidence > 0.7:
            return "🔴 HIGH"
        elif self.confidence > 0.5:
            return "🟠 MEDIUM"
        elif self.confidence > 0.3:
            return "🟡 LOW"
        else:
            return "🟢 MINIMAL"
    
    def _generate_summary(self, ticker: str) -> str:
        """Generate summary."""
        risk_level = self._calculate_risk_level()
        
        return f"""🐻 BEAR AGENT REPORT - {ticker}
Risk Level: {risk_level}
Recommendation: {self.arguments[0]['reason'] if self.arguments else 'No risks found'}
Detected Risks: {len(self.arguments)} items"""

    def debate_opening(self) -> str:
        """Debate opening statement."""
        if not self.arguments:
            return "No risks detected."
        
        opening = f"🐻 BEAR POSITION - RISKS:\n"
        for i, arg in enumerate(self.arguments[:3], 1):
            opening += f"{i}. ⚠️ {arg['reason']}\n"
        
        return opening
