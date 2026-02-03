"""
Bull Agent - Bullish Market Sentiment Analyzer
Author: Erdinc Erdogan
Purpose: Identifies bullish opportunities and upside scenarios in financial markets by analyzing
technical indicators, macro factors, and sentiment data to advocate buy positions.
References:
- Technical Analysis Indicators (RSI, Golden Cross, Volume Surge)
- Momentum Trading Strategies
- Sector Rotation and Capital Flow Analysis
Usage:
    bull = BullAgent(llm=ollama_llm)
    result = bull.analyze(ticker="AAPL", price_data=data, technicals=tech_signals)
"""
import os
from typing import Dict, List, Optional
from datetime import datetime
from colorama import Fore, Style


class BullAgent:
    """
    Bull Agent.
    
    Task: Identify and defend bullish scenarios in the market.
    Always advocates "BUY" positions and presents reasons.
    """
    
    # Bullish indicators
    BULLISH_INDICATORS = {
        "rsi_oversold": {"condition": "rsi < 30", "weight": 0.8, "reason": "RSI is in the oversold zone - a rebound is expected"},
        "golden_cross": {"condition": "ma50 > ma200", "weight": 0.9, "reason": "Golden cross - strong bullish signal"},
        "volume_surge": {"condition": "volume > avg_volume * 2", "weight": 0.7, "reason": "Volume surge - buyer interest"},
        "support_bounce": {"condition": "price near support", "weight": 0.75, "reason": "Support level holding"},
        "bullish_divergence": {"condition": "price down, rsi up", "weight": 0.85, "reason": "Bullish divergence"},
        "earnings_beat": {"condition": "eps > expected", "weight": 0.8, "reason": "Earnings above expectations"},
        "insider_buying": {"condition": "net_insider > 0", "weight": 0.7, "reason": "Insider buying"},
        "sector_rotation": {"condition": "sector_inflow", "weight": 0.6, "reason": "Capital inflow to sector"},
    }
    
    # Macro bullish factors
    MACRO_BULLISH = [
        "fed_dovish",     # FED dovish
        "inflation_cooling",  # Inflation cooling
        "employment_strong",  # Strong employment
        "gdp_growth",     # GDP growth
        "ai_hype",        # AI hype
    ]
    
    def __init__(self, llm=None):
        """
        Args:
            llm: LangChain LLM instance
        """
        self.llm = llm
        self.name = "🐂 BULL AGENT"
        self.bias = "BULLISH"
        self.confidence = 0.0
        self.arguments = []
    
    def analyze(self, 
                ticker: str,
                price_data: Dict,
                news: List[str] = None,
                technicals: Dict = None) -> Dict:
        """
        Analyze from a bullish perspective.
        
        Args:
            ticker: Stock symbol
            price_data: Price data
            news: Recent news
            technicals: Technical indicators
        """
        print(f"{Fore.GREEN}{self.name}: Analyzing {ticker}...{Style.RESET_ALL}", flush=True)
        
        self.arguments = []
        total_weight = 0
        max_weight = 0
        
        tech = technicals or {}
        
        # Check technical indicators
        # RSI
        rsi = tech.get("rsi", 50)
        if rsi < 30:
            self._add_argument("rsi_oversold", rsi)
            total_weight += 0.8
        elif rsi < 40:
            self._add_argument("rsi_low", rsi, "RSI is low - potential rebound")
            total_weight += 0.4
        
        # Trend
        trend = tech.get("trend", "").lower()
        if "bull" in trend or "yukselis" in trend.lower() if trend else False:
            self._add_argument("trend_bullish", trend, "Bullish trend active")
            total_weight += 0.6
        
        # Price momentum
        price = price_data.get("close", price_data.get("price", 0))
        prev_price = price_data.get("prev_close", price * 0.99)
        if price > prev_price:
            change_pct = ((price - prev_price) / prev_price) * 100
            self._add_argument("positive_momentum", change_pct, f"Positive momentum: +%{change_pct:.2f}")
            total_weight += 0.3
        
        # Volume
        volume = price_data.get("volume", 0)
        avg_volume = price_data.get("avg_volume", volume)
        if volume > avg_volume * 1.5:
            self._add_argument("volume_surge", volume, "Volume surge - increased buyer interest")
            total_weight += 0.5
        
        # FVG (Fair Value Gap) analysis
        fvgs = tech.get("fvgs", [])
        bullish_fvgs = [f for f in fvgs if f.get("type", "").lower() == "bullish"]
        if bullish_fvgs:
            self._add_argument("bullish_fvg", len(bullish_fvgs), f"{len(bullish_fvgs)} bullish FVG found")
            total_weight += 0.4
        
        # News analysis
        if news:
            bullish_news = self._analyze_news_sentiment(news)
            if bullish_news:
                self._add_argument("positive_news", bullish_news, f"Positive news: {len(bullish_news)}")
                total_weight += 0.3 * len(bullish_news)
        
        # LLM deep analysis
        if self.llm:
            llm_analysis = self._llm_bullish_analysis(ticker, price_data, technicals)
            if llm_analysis:
                self.arguments.append(llm_analysis)
                total_weight += 0.5
        
        # Calculate confidence
        max_weight = len(self.BULLISH_INDICATORS) * 0.8
        self.confidence = min(total_weight / max_weight, 1.0) if max_weight > 0 else 0.5
        
        # Result
        result = {
            "agent": self.name,
            "ticker": ticker,
            "recommendation": "BUY" if self.confidence > 0.5 else "HOLD",
            "confidence": self.confidence,
            "bias": self.bias,
            "arguments": self.arguments,
            "summary": self._generate_summary(ticker)
        }
        
        print(f"{Fore.GREEN}{self.name}: {result['recommendation']} (Confidence: %{self.confidence*100:.0f}){Style.RESET_ALL}", flush=True)
        
        return result
    
    def _add_argument(self, indicator: str, value, reason: str = None):
        """Add argument."""
        if reason is None:
            reason = self.BULLISH_INDICATORS.get(indicator, {}).get("reason", indicator)
        
        self.arguments.append({
            "indicator": indicator,
            "value": value,
            "reason": reason,
            "sentiment": "BULLISH"
        })
    
    def _analyze_news_sentiment(self, news: List[str]) -> List[str]:
        """Search for bullish signals in news."""
        bullish_keywords = [
            "growth", "beat", "surge", "rally", "upgrade", "buy",
            "breakout", "record", "strong", "exceeds", "outperform",
            "bullish", "upside", "innovation", "expansion"
        ]
        
        bullish_news = []
        for article in news:
            article_lower = article.lower()
            if any(kw in article_lower for kw in bullish_keywords):
                bullish_news.append(article)
        
        return bullish_news[:3]  # Max 3 news
    
    def _llm_bullish_analysis(self, ticker: str, price_data: Dict, technicals: Dict) -> Optional[Dict]:
        """LLM bullish analysis."""
        if not self.llm:
            return None
        
        prompt = f"""You are a BULL analyst. Analyze ONLY bullish scenarios for the {ticker} stock.

Technical Data:
- Price: ${price_data.get('price', price_data.get('close', 0)):.2f}
- RSI: {technicals.get('rsi', 'N/A')}
- Trend: {technicals.get('trend', 'N/A')}

Task: Find the top 2-3 strongest bullish arguments. Look only from a positive perspective.
Format: Short and concise, bullet points."""

        try:
            response = self.llm.invoke(prompt)
            return {
                "indicator": "llm_analysis",
                "value": response[:500],
                "reason": "AI Deep Bullish Analysis",
                "sentiment": "BULLISH"
            }
        except:
            return None
    
    def _generate_summary(self, ticker: str) -> str:
        """Generate summary."""
        if self.confidence > 0.7:
            strength = "STRONG"
        elif self.confidence > 0.5:
            strength = "MEDIUM"
        else:
            strength = "WEAK"
        
        return f"""🐂 BULL AGENT REPORT - {ticker}
Recommendation: {'BUY ✅' if self.confidence > 0.5 else 'HOLD ⏳'}
Confidence: %{self.confidence*100:.0f} ({strength})
Number of Arguments: {len(self.arguments)}
Strongest Argument: {self.arguments[0]['reason'] if self.arguments else 'None'}"""

    def debate_opening(self) -> str:
        """Debate opening statement."""
        if not self.arguments:
            return "Analysis not yet performed."
        
        opening = f"🐂 BULL POSITION (%{self.confidence*100:.0f} confidence):\n"
        for i, arg in enumerate(self.arguments[:3], 1):
            opening += f"{i}. {arg['reason']}\n"
        
        return opening
