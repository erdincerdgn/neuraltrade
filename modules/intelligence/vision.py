"""
Vision-RAG for Chart Pattern Analysis
Author: Erdinc Erdogan
Purpose: Analyzes trading chart images using vision LLMs (LLaVA, Llama3.2-Vision) to detect patterns, support/resistance, and entry/exit points.
References:
- LLaVA (Large Language-and-Vision Assistant)
- Chart Pattern Recognition
- mplfinance Visualization
Usage:
    vision = VisionRAG(ollama_host="http://localhost:11434")
    analysis = vision.analyze_chart(image_path="chart.png", query="What patterns do you see?")
"""
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from colorama import Fore, Style


class VisionRAG:
    """Grafik görsellerini analiz eder."""
    
    def __init__(self, ollama_host: str):
        self.ollama_host = ollama_host
        self.vision_model = "llama3.2-vision"
        self.available = False
        self._check_availability()
    
    def _check_availability(self):
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.ollama_host}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as response:
                data = response.read().decode('utf-8')
                self.available = "vision" in data.lower() or "llava" in data.lower()
        except:
            self.available = False
    
    def analyze_chart(self, image_path: str, query: str = "Bu grafikte hangi formasyonlar var?") -> str:
        if not self.available:
            return "<vision_analysis>Vision model mevcut değil</vision_analysis>"
        
        print(f"{Fore.MAGENTA}  → Vision-RAG: Grafik analiz ediliyor...{Style.RESET_ALL}", flush=True)
        
        try:
            import base64
            import urllib.request
            import json
            
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            payload = json.dumps({
                "model": self.vision_model,
                "prompt": f"""Bu trading grafiğini analiz et:
{query}

Özellikle şunları kontrol et:
1. Chart patterns (OBO, Double Top/Bottom, Triangle, vb.)
2. Support/Resistance seviyeleri
3. Trend yönü
4. Entry/Exit önerileri""",
                "images": [image_data]
            }).encode('utf-8')
            
            req = urllib.request.Request(
                f"{self.ollama_host}/api/generate",
                data=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req, timeout=30) as response:
                result = response.read().decode('utf-8')
                lines = result.strip().split('\n')
                full_response = ""
                for line in lines:
                    try:
                        data = json.loads(line)
                        full_response += data.get("response", "")
                    except:
                        pass
                
                if full_response:
                    print(f"{Fore.GREEN}  → Vision-RAG: Analiz tamamlandı{Style.RESET_ALL}", flush=True)
                    return f"<vision_analysis>\n👁️ GRAFİK ANALİZİ:\n{full_response}\n</vision_analysis>\n"
        except Exception as e:
            print(f"{Fore.RED}  → Vision-RAG hatası: {e}{Style.RESET_ALL}", flush=True)
        
        return ""
    
    def capture_chart_screenshot(self, ticker: str, price_data: Optional[any] = None) -> Optional[str]:
        """mplfinance ile candlestick grafiği oluştur."""
        chart_path = f"/app/data/charts/{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        try:
            import mplfinance as mpf
            import pandas as pd
            
            if price_data is None or price_data.empty:
                return None
            
            df = price_data.copy()
            if 'open' in df.columns:
                df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
            
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            Path(chart_path).parent.mkdir(parents=True, exist_ok=True)
            
            style = mpf.make_mpf_style(base_mpf_style='nightclouds', gridstyle='', rc={'font.size': 8})
            
            mpf.plot(
                df.tail(50),
                type='candle',
                style=style,
                title=f'{ticker} - Price Action',
                ylabel='Price ($)',
                volume=True if 'Volume' in df.columns else False,
                savefig=dict(fname=chart_path, dpi=150, bbox_inches='tight'),
                figratio=(16, 9)
            )
            
            print(f"{Fore.GREEN}  → Vision: Grafik oluşturuldu: {chart_path}{Style.RESET_ALL}", flush=True)
            return chart_path
        except:
            pass
        
        return None
