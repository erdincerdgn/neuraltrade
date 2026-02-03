"""
AI Advisor Backup (Legacy Hedge Fund Features)
Author: Erdinc Erdogan
Purpose: Backup of advanced AI advisor with multi-modal vision, property graph, model orchestrator, agentic tool use, and C-RAG with web fallback.
References:
- Multi-Modal Vision (Chart Analysis)
- Agentic Tool Use (LLM Planning)
- Chain-of-Thought (8-step reasoning)
- HyDE + Multi-Query RAG
Usage:
    # Legacy backup - see modules/intelligence/advisor.py for current version
    advisor = AIAdvisor()
    response = advisor.analyze(query, ticker)
"""

import os
import re
import sqlite3
import traceback
from io import StringIO
from contextlib import redirect_stdout
from typing import List, Dict, Optional, Tuple
from enum import Enum
from datetime import datetime
from pathlib import Path
from colorama import Fore, Style
from qdrant_client import QdrantClient, models

from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from flashrank import Ranker, RerankRequest

# Cross-Encoder için
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

# yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Python REPL için
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


# ============================================
# ENUMS
# ============================================
class QueryType(Enum):
    TRADE = "trade"
    TECHNICAL = "tech"
    GENERAL = "general"


class QueryComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


# ============================================
# A. SQLITE MEMORY (Thread-Safe)
# ============================================
class SQLiteMemory:
    """
    SQLite tabanlı kullanıcı hafızası.
    JSON'a göre daha güvenli ve eşzamanlı kullanıcı desteği sağlar.
    """
    
    def __init__(self, db_path: str = "/app/data/user_memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Veritabanı tablolarını oluştur."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Users tablosu
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                risk_tolerance TEXT DEFAULT 'moderate',
                trading_style TEXT DEFAULT 'swing',
                leverage_pref TEXT DEFAULT 'low',
                successful_trades INTEGER DEFAULT 0,
                total_trades INTEGER DEFAULT 0,
                created_at TEXT,
                last_active TEXT
            )
        ''')
        
        # Recommendations tablosu
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                ticker TEXT,
                action TEXT,
                entry_price REAL,
                confidence INTEGER,
                timestamp TEXT,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')
        
        # Backtest Results tablosu (Deneyimden Öğrenme için)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recommendation_id INTEGER,
                exit_price REAL,
                profit_loss REAL,
                profit_pct REAL,
                outcome TEXT,
                lesson_learned TEXT,
                timestamp TEXT,
                FOREIGN KEY (recommendation_id) REFERENCES recommendations(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_user_profile(self, user_id: str = "default") -> Dict:
        """Kullanıcı profilini getir."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        
        if not row:
            # Yeni kullanıcı oluştur
            now = datetime.now().isoformat()
            cursor.execute('''
                INSERT INTO users (user_id, created_at, last_active) VALUES (?, ?, ?)
            ''', (user_id, now, now))
            conn.commit()
            row = (user_id, 'moderate', 'swing', 'low', 0, 0, now, now)
        
        conn.close()
        
        return {
            "user_id": row[0],
            "risk_tolerance": row[1],
            "trading_style": row[2],
            "leverage_pref": row[3],
            "successful_trades": row[4],
            "total_trades": row[5],
            "created_at": row[6],
            "last_active": row[7]
        }
    
    def update_preference(self, user_id: str, key: str, value: str):
        """Kullanıcı tercihini güncelle."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Kullanıcı var mı kontrol et
        self.get_user_profile(user_id)
        
        # Güncelle
        cursor.execute(f'''
            UPDATE users SET {key} = ?, last_active = ? WHERE user_id = ?
        ''', (value, datetime.now().isoformat(), user_id))
        
        conn.commit()
        conn.close()
    
    def add_recommendation(self, user_id: str, ticker: str, action: str, confidence: int):
        """Öneriyi kaydet."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO recommendations (user_id, ticker, action, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, ticker, action, confidence, datetime.now().isoformat()))
        
        cursor.execute('''
            UPDATE users SET total_trades = total_trades + 1, last_active = ?
            WHERE user_id = ?
        ''', (datetime.now().isoformat(), user_id))
        
        conn.commit()
        conn.close()
    
    def get_last_recommendations(self, user_id: str, limit: int = 5) -> List[Dict]:
        """Son önerileri getir."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT ticker, action, confidence, timestamp FROM recommendations
            WHERE user_id = ? ORDER BY id DESC LIMIT ?
        ''', (user_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [{"ticker": r[0], "action": r[1], "confidence": r[2], "timestamp": r[3]} for r in rows]
    
    def get_personalization_context(self, user_id: str) -> str:
        """Kişiselleştirme bağlamı oluştur."""
        profile = self.get_user_profile(user_id)
        recs = self.get_last_recommendations(user_id, 3)
        
        context = f"""
═══════════════════════════════════════════════════════════
👤 KULLANICI PROFİLİ (SQLite Memory)
═══════════════════════════════════════════════════════════
• Risk Toleransı: {profile['risk_tolerance'].upper()}
• Trading Stili: {profile['trading_style'].upper()}
• Kaldıraç: {profile['leverage_pref'].upper()}
• Toplam Analiz: {profile['total_trades']}
"""
        if recs:
            context += f"• Son Öneri: {recs[0]['ticker']} - {recs[0]['action']} ({recs[0]['confidence']}%)\n"
        
        return context
    
    def record_trade_result(self, recommendation_id: int, exit_price: float, entry_price: float, outcome: str, lesson: str):
        """Trade sonucunu kaydet (Backtest Learning için)."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        profit_loss = exit_price - entry_price
        profit_pct = (profit_loss / entry_price) * 100 if entry_price > 0 else 0
        
        cursor.execute('''
            INSERT INTO backtest_results (recommendation_id, exit_price, profit_loss, profit_pct, outcome, lesson_learned, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (recommendation_id, exit_price, profit_loss, profit_pct, outcome, lesson, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def get_lessons_learned(self, user_id: str, limit: int = 5) -> List[Dict]:
        """Geçmiş trade'lerden çıkarılan dersleri getir."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT r.ticker, r.action, r.confidence, b.profit_pct, b.outcome, b.lesson_learned
            FROM backtest_results b
            JOIN recommendations r ON b.recommendation_id = r.id
            WHERE r.user_id = ?
            ORDER BY b.id DESC LIMIT ?
        ''', (user_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [{"ticker": r[0], "action": r[1], "confidence": r[2], 
                 "profit_pct": r[3], "outcome": r[4], "lesson": r[5]} for r in rows]
    
    def analyze_past_performance(self, user_id: str) -> str:
        """Geçmiş performansı analiz et ve öğrenilen dersleri özetle."""
        lessons = self.get_lessons_learned(user_id, 10)
        
        if not lessons:
            return ""
        
        wins = sum(1 for l in lessons if l["outcome"] == "WIN")
        losses = len(lessons) - wins
        avg_profit = sum(l["profit_pct"] for l in lessons) / len(lessons) if lessons else 0
        
        context = f"""
<backtest_learning>
📈 GEÇMİŞ PERFORMANS ANALİZİ:
• Kazanç/Kayıp: {wins}W / {losses}L
• Ortalama Kar: {avg_profit:.2f}%
"""
        # Son dersler
        if lessons:
            context += "• Son Dersler:\n"
            for l in lessons[:3]:
                if l["lesson"]:
                    context += f"  - {l['ticker']}: {l['lesson']}\n"
        
        context += "</backtest_learning>\n"
        return context


# ============================================
# B. PYTHON REPL (Hesaplama Motoru)
# ============================================
class PythonREPL:
    """
    Python kod çalıştırıcı.
    Finansal hesaplamalar için kullanılır (pip, kaldıraç, risk).
    """
    
    @staticmethod
    def execute(code: str) -> Tuple[bool, str]:
        """Python kodunu çalıştır ve sonucu döndür."""
        # Güvenlik: Tehlikeli işlemleri engelle
        forbidden = ['import os', 'import sys', 'subprocess', 'eval(', 'exec(', 'open(', '__import__']
        for f in forbidden:
            if f in code:
                return False, f"Güvenlik hatası: {f} kullanılamaz"
        
        # Çalıştır
        try:
            # Output yakalama
            output = StringIO()
            
            # Güvenli namespace
            safe_globals = {
                "pd": pd if PANDAS_AVAILABLE else None,
                "np": np if PANDAS_AVAILABLE else None,
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "len": len,
                "print": lambda *args: output.write(' '.join(map(str, args)) + '\n'),
            }
            
            exec(code, safe_globals)
            result = output.getvalue()
            
            return True, result.strip() if result.strip() else "Hesaplama tamamlandı."
        except Exception as e:
            return False, f"Hata: {str(e)}"
    
    @staticmethod
    def calculate_pip_value(lot_size: float, pip_size: float = 0.0001, account_currency: str = "USD") -> str:
        """Pip değeri hesapla."""
        pip_value = lot_size * pip_size * 100000  # Standard lot
        return f"Pip Değeri: ${pip_value:.2f} ({lot_size} lot için)"
    
    @staticmethod
    def calculate_position_size(account_balance: float, risk_percent: float, stop_loss_pips: float, pip_value: float = 10) -> str:
        """Pozisyon büyüklüğü hesapla."""
        risk_amount = account_balance * (risk_percent / 100)
        position_size = risk_amount / (stop_loss_pips * pip_value)
        return f"Pozisyon: {position_size:.2f} lot (Risk: ${risk_amount:.2f})"
    
    @staticmethod
    def calculate_risk_reward(entry: float, stop_loss: float, take_profit: float) -> str:
        """Risk/Reward oranı hesapla."""
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        ratio = reward / risk if risk > 0 else 0
        return f"Risk/Reward: 1:{ratio:.2f} (Risk: {risk:.4f}, Reward: {reward:.4f})"


# ============================================
# C. CORRECTIVE RAG (C-RAG)
# ============================================
class CorrectiveRAG:
    """
    Döküman alaka kontrolü.
    Alakasız dökümanları atar, gerekirse web araması yapar.
    """
    
    def __init__(self, llm: OllamaLLM):
        self.llm = llm
    
    def check_relevance(self, query: str, doc: Document) -> Tuple[bool, float]:
        """Dökümanın sorguyla alakalı olup olmadığını kontrol et."""
        prompt = f"""Bu döküman, aşağıdaki soruyla ALAKALI mı?
Sadece 0-100 arası bir skor yaz.

SORU: {query[:200]}
DÖKÜMAN: {doc.page_content[:300]}

ALAKA SKORU (0-100):"""

        try:
            response = self.llm.invoke(prompt)
            score = int(''.join(filter(str.isdigit, response[:5])))
            return score >= 25, min(score, 100)  # Threshold lowered to 25
        except:
            return True, 70  # Default: alakalı kabul et
    
    def filter_documents(self, query: str, docs: List[Document], min_score: int = 25) -> List[Document]:
        """Alakasız dökümanları filtrele - DİNAMİK EŞİK."""
        if not docs:
            return docs
        
        # Tüm skorları topla
        scored_docs = []
        for doc in docs:
            is_relevant, score = self.check_relevance(query, doc)
            doc.metadata["relevance_score"] = score
            scored_docs.append((doc, score))
        
        # Dinamik eşik: Ortalama skorun %60'ı veya min_score
        avg_score = sum(s for _, s in scored_docs) / len(scored_docs) if scored_docs else 0
        dynamic_threshold = max(min_score, avg_score * 0.6)
        
        # Filtrele
        filtered = [doc for doc, score in scored_docs if score >= dynamic_threshold]
        
        print(f"{Fore.CYAN}    Dinamik Eşik: {dynamic_threshold:.0f} (Ort: {avg_score:.0f}){Style.RESET_ALL}", flush=True)
        
        # Fallback: Hiç döküman geçmediyse, en az ilk 2'yi döndür
        if not filtered and docs:
            return docs[:2]
        
        return filtered
    
    def web_search_fallback(self, ticker: str, query: str) -> str:
        """
        Dökümanlar alakasız ise web'den son haberleri çek.
        DuckDuckGo üzerinden basit arama.
        """
        print(f"{Fore.YELLOW}  → Web Fallback: {ticker} haberleri aranıyor...{Style.RESET_ALL}", flush=True)
        
        try:
            # DuckDuckGo HTML search (API gerektirmez)
            import urllib.request
            import urllib.parse
            
            search_query = urllib.parse.quote(f"{ticker} stock news today")
            url = f"https://html.duckduckgo.com/html/?q={search_query}"
            
            headers = {'User-Agent': 'Mozilla/5.0'}
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=5) as response:
                html = response.read().decode('utf-8')
            
            # Basit ayrıştırma - başlıkları çek
            import re
            titles = re.findall(r'class="result__a"[^>]*>([^<]+)</a>', html)
            snippets = re.findall(r'class="result__snippet"[^>]*>([^<]+)', html)
            
            if titles:
                news = []
                for i, (title, snippet) in enumerate(zip(titles[:3], snippets[:3])):
                    news.append(f"• {title.strip()}")
                
                result = "\n".join(news)
                print(f"{Fore.GREEN}  → Web Fallback: {len(news)} haber bulundu{Style.RESET_ALL}", flush=True)
                return f"\n🌐 SON HABERLER ({ticker}):\n{result}\n"
        except Exception as e:
            print(f"{Fore.RED}  → Web Fallback hatası: {e}{Style.RESET_ALL}", flush=True)
        
        return ""


# ============================================
# E. ENTRY/EXIT LEVEL EXTRACTOR
# ============================================
class TradeRuleExtractor:
    """
    Dökümanlardan somut trade kuralları ve seviyeleri çıkarır.
    "RSI < 30 ise AL", "Support: 265.50" gibi kuralları tespit eder.
    """
    
    def __init__(self, llm: OllamaLLM):
        self.llm = llm
    
    def extract_trade_rules(self, docs: List[Document], ticker: str, current_price: float) -> str:
        """Dökümanlardan trade kuralları çıkarır."""
        if not docs:
            return ""
        
        # Tüm döküman içeriğini birleştir
        content = "\n".join([doc.page_content[:500] for doc in docs[:3]])
        
        prompt = f"""Aşağıdaki trading dökümanlarından somut TRADE KURALLARI çıkar.

DÖKÜMANLAR:
{content[:2000]}

MEVCUT BİLGİLER:
- Sembol: {ticker}
- Fiyat: ${current_price}

GÖREV: Şu formatta listele:
1. ENTRY KURALI: [koşul ve seviye]
2. STOP-LOSS: [seviye veya yüzde]
3. TAKE-PROFIT: [seviye veya yüzde]
4. ÖZEL KOŞULLAR: [varsa]

Sadece dökümanlarda geçen bilgileri kullan. Yoksa "Belirtilmemiş" yaz.

KURALLAR:"""

        try:
            response = self.llm.invoke(prompt)
            if response.strip():
                return f"\n📍 SOMUT TRADE KURALLARI:\n{response.strip()}\n"
        except:
            pass
        
        return ""
    
    def calculate_levels(self, current_price: float, rsi: float) -> str:
        """RSI ve fiyata göre otomatik seviyeler hesapla."""
        # Basit seviye hesaplama
        if rsi < 30:
            # Oversold - AL sinyali
            entry = current_price
            stop_loss = current_price * 0.98  # %2 altı
            take_profit = current_price * 1.03  # %3 üstü
            signal = "AL 🟢"
        elif rsi > 70:
            # Overbought - SAT sinyali
            entry = current_price
            stop_loss = current_price * 1.02  # %2 üstü
            take_profit = current_price * 0.97  # %3 altı
            signal = "SAT 🔴"
        else:
            # Nötr - BEKLE
            entry = current_price
            stop_loss = current_price * 0.97
            take_profit = current_price * 1.03
            signal = "BEKLE 🟡"
        
        return f"""
📊 OTOMATİK SEVİYE HESAPLAMASI (RSI: {rsi:.1f}):
• Sinyal: {signal}
• Entry: ${entry:.2f}
• Stop-Loss: ${stop_loss:.2f} ({((stop_loss/entry)-1)*100:.1f}%)
• Take-Profit: ${take_profit:.2f} ({((take_profit/entry)-1)*100:.1f}%)
• Risk/Reward: 1:{abs((take_profit-entry)/(entry-stop_loss)):.1f}
"""


# ============================================
# F. ACTIVE TOOL USE (Agentic Decision Making)
# ============================================
class AgenticToolUse:
    """
    LLM'in araçları dinamik olarak seçmesini sağlar.
    Statik pipeline yerine LLM soruya göre araç sırası belirler.
    """
    
    AVAILABLE_TOOLS = {
        "web_search": "Web'den güncel haberler ara",
        "calculate": "Pip/Kaldıraç/Risk hesapla",
        "document_search": "Kitaplardan strateji ara",
        "chart_analyze": "Grafik formasyonlarını analiz et",
        "fact_check": "Fiyatı canlı doğrula"
    }
    
    def __init__(self, llm: OllamaLLM):
        self.llm = llm
    
    def plan_tools(self, query: str) -> List[str]:
        """Sorguya göre hangi araçların hangi sırayla kullanılacağını belirle."""
        tools_list = "\n".join([f"- {k}: {v}" for k, v in self.AVAILABLE_TOOLS.items()])
        
        prompt = f"""Aşağıdaki soruyu cevaplamak için hangi araçları hangi sırayla kullanmalıyım?

SORU: {query}

MEVCUT ARAÇLAR:
{tools_list}

Sadece araç isimlerini virgülle ayırarak yaz (örn: document_search, calculate, fact_check):"""

        try:
            response = self.llm.invoke(prompt)
            tools = [t.strip().lower() for t in response.split(",")]
            # Geçerli araçları filtrele
            valid_tools = [t for t in tools if t in self.AVAILABLE_TOOLS]
            
            if valid_tools:
                return valid_tools
        except:
            pass
        
        # Default tool sequence
        return ["document_search", "fact_check", "calculate"]
    
    def execute_tool(self, tool_name: str, context: Dict) -> str:
        """Belirtilen aracı çalıştır."""
        # Bu method AIAdvisor'da implement edilecek
        return f"[{tool_name} executed]"


# ============================================
# G. VISION-RAG (Grafik Analizi)
# ============================================
class VisionRAG:
    """
    Grafik görsellerini analiz eder.
    Chart patterns (OBO, Double Top, vb.) tespit eder.
    Not: Llama 3.2-Vision veya benzeri model gerektirir.
    """
    
    def __init__(self, ollama_host: str):
        self.ollama_host = ollama_host
        self.vision_model = "llama3.2-vision"  # Veya gpt-4o-mini
        self.available = False
        self._check_availability()
    
    def _check_availability(self):
        """Vision model mevcut mu kontrol et."""
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.ollama_host}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as response:
                data = response.read().decode('utf-8')
                self.available = "vision" in data.lower() or "llava" in data.lower()
        except:
            self.available = False
    
    def analyze_chart(self, image_path: str, query: str = "Bu grafikte hangi formasyonlar var?") -> str:
        """Grafik görselini analiz et."""
        if not self.available:
            return "<vision_analysis>Vision model mevcut değil</vision_analysis>"
        
        print(f"{Fore.MAGENTA}  → Vision-RAG: Grafik analiz ediliyor...{Style.RESET_ALL}", flush=True)
        
        # Base64 encode image
        try:
            import base64
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Ollama vision API call
            import urllib.request
            import json
            
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
                # Parse streaming response
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
        """
        mplfinance ile candlestick grafiği oluştur.
        """
        chart_path = f"/app/data/charts/{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        try:
            import mplfinance as mpf
            import pandas as pd
            
            # price_data DataFrame olmalı (Open, High, Low, Close, Volume)
            if price_data is None or price_data.empty:
                print(f"{Fore.YELLOW}  → Vision: Grafik verisi yok{Style.RESET_ALL}", flush=True)
                return None
            
            # Ensure proper column names
            df = price_data.copy()
            if 'open' in df.columns:
                df = df.rename(columns={
                    'open': 'Open', 'high': 'High', 'low': 'Low', 
                    'close': 'Close', 'volume': 'Volume'
                })
            
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Create chart directory
            Path(chart_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Generate chart with custom style
            style = mpf.make_mpf_style(
                base_mpf_style='nightclouds',
                gridstyle='',
                rc={'font.size': 8}
            )
            
            # Save chart
            mpf.plot(
                df.tail(50),  # Son 50 mum
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
            
        except ImportError:
            print(f"{Fore.YELLOW}  → Vision: mplfinance yüklü değil{Style.RESET_ALL}", flush=True)
        except Exception as e:
            print(f"{Fore.RED}  → Vision grafik hatası: {e}{Style.RESET_ALL}", flush=True)
        
        return None
    
    def analyze_with_data(self, ticker: str, price_data: any, query: str = "Bu grafikte hangi formasyonlar var?") -> str:
        """Veri ile grafik oluştur ve analiz et."""
        chart_path = self.capture_chart_screenshot(ticker, price_data)
        if chart_path:
            return self.analyze_chart(chart_path, query)
        return ""


# ============================================
# D. CROSS-ENCODER RERANKING
# ============================================
class CrossEncoderReranker:
    """
    Cross-Encoder ile ultra hassas sıralama.
    Bi-Encoder'dan daha yavaş ama çok daha doğru.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        if CROSS_ENCODER_AVAILABLE:
            try:
                self.model = CrossEncoder(model_name)
                self.available = True
            except:
                self.available = False
        else:
            self.available = False
    
    def rerank(self, query: str, docs: List[Document], top_k: int = 3) -> List[Document]:
        """Dökümanları cross-encoder ile yeniden sırala."""
        if not self.available or not docs:
            return docs[:top_k]
        
        # Query-Doc pairs oluştur
        pairs = [[query, doc.page_content[:500]] for doc in docs]
        
        # Score hesapla
        scores = self.model.predict(pairs)
        
        # Sort by score
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in scored_docs[:top_k]]


# ============================================
# GRAPH-RAG & ROUTER & ADAPTIVE
# ============================================
class KnowledgeGraph:
    RELATIONS = {
        ("faiz_artisi", "causes", "usd_guclenme"): 0.9,
        ("faiz_artisi", "causes", "altin_dusus"): 0.7,
        ("enflasyon_artisi", "causes", "faiz_artisi"): 0.8,
        ("usd_guclenme", "causes", "emtia_dusus"): 0.7,
        ("rsi_oversold", "suggests", "olasi_toparlanma"): 0.6,
    }
    
    CONCEPT_MAPPING = {
        "faiz": "faiz_artisi", "enflasyon": "enflasyon_artisi",
        "dolar": "usd_guclenme", "rsi": "rsi_oversold"
    }
    
    @classmethod
    def get_graph_context(cls, query: str) -> str:
        q = query.lower()
        found = [c for kw, c in cls.CONCEPT_MAPPING.items() if kw in q]
        relations = [(s, r, t, c) for (s, r, t), c in cls.RELATIONS.items() if s in found]
        
        if not relations:
            return ""
        
        ctx = "\n🕸️ EKONOMİK İLİŞKİLER:\n"
        for s, r, t, c in relations:
            ctx += f"  • {s.replace('_', ' ').title()} → {t.replace('_', ' ').title()} ({int(c*100)}%)\n"
        return ctx


class SemanticRouter:
    TRADE_KW = ["trade", "al", "sat", "strateji", "rsi", "macd", "forex"]
    GENERAL_KW = ["merhaba", "selam", "nasılsın"]
    
    @classmethod
    def classify(cls, q: str) -> QueryType:
        q = q.lower()
        if any(kw in q for kw in cls.GENERAL_KW):
            return QueryType.GENERAL
        if sum(1 for kw in cls.TRADE_KW if kw in q) >= 1:
            return QueryType.TRADE
        return QueryType.TECHNICAL


class AdaptiveRAG:
    @classmethod
    def assess(cls, q: str) -> QueryComplexity:
        q = q.lower()
        if any(x in q for x in ["nasıl etkilenir", "ilişkisi", "karşılaştır"]):
            return QueryComplexity.COMPLEX
        if any(x in q for x in ["nedir", "ne demek"]):
            return QueryComplexity.SIMPLE
        return QueryComplexity.MODERATE
    
    @classmethod
    def get_strategy(cls, c: QueryComplexity) -> Dict:
        return {
            QueryComplexity.SIMPLE: {"use_crag": False, "use_cross": False, "top_k": 3, "desc": "⚡ HIZLI"},
            QueryComplexity.MODERATE: {"use_crag": True, "use_cross": False, "top_k": 5, "desc": "📊 STANDART"},
            QueryComplexity.COMPLEX: {"use_crag": True, "use_cross": True, "top_k": 10, "desc": "🔬 AI LAB"}
        }.get(c)


# ============================================
# H. PROPERTY GRAPH (Gelişmiş Graph-RAG)
# ============================================
class PropertyGraph:
    """
    Dökümanlardan otomatik ilişki çıkarır.
    Apple → iPhone Satışları → Çip Tedarikçileri gibi bağlantılar kurar.
    """
    
    # Sektör ve şirket ilişkileri
    SECTOR_RELATIONS = {
        "AAPL": ["technology", "consumer_electronics", "TSMC", "QCOM", "semiconductors"],
        "MSFT": ["technology", "cloud", "AI", "enterprise"],
        "GOOGL": ["technology", "advertising", "AI", "cloud"],
        "NVDA": ["semiconductors", "AI", "gaming", "datacenter"],
        "TSLA": ["automotive", "energy", "batteries", "lithium"],
        "AMZN": ["ecommerce", "cloud", "logistics", "retail"],
    }
    
    # Ripple effect kuralları
    RIPPLE_EFFECTS = {
        "semiconductor_shortage": {"AAPL": -0.8, "TSLA": -0.6, "NVDA": -0.5},
        "ai_boom": {"NVDA": 0.9, "MSFT": 0.7, "GOOGL": 0.7},
        "interest_rate_hike": {"growth_stocks": -0.6, "banks": 0.5},
        "oil_price_increase": {"energy": 0.7, "transportation": -0.5},
        "china_tension": {"AAPL": -0.4, "semiconductors": -0.6},
    }
    
    def __init__(self, llm: OllamaLLM = None):
        self.llm = llm
        self.extracted_relations = {}
    
    def extract_relations_from_doc(self, doc: Document) -> List[Tuple[str, str, str]]:
        """Döküman içeriğinden ilişkileri otomatik çıkar."""
        relations = []
        content = doc.page_content.lower()
        
        # Basit pattern matching
        patterns = [
            (r"(\w+) increases? (\w+)", "increases"),
            (r"(\w+) decreases? (\w+)", "decreases"),
            (r"(\w+) affects? (\w+)", "affects"),
            (r"(\w+) leads? to (\w+)", "leads_to"),
            (r"(\w+) causes? (\w+)", "causes"),
        ]
        
        for pattern, relation in patterns:
            matches = re.findall(pattern, content)
            for source, target in matches[:5]:  # Max 5 per pattern
                relations.append((source, relation, target))
        
        return relations
    
    def get_ripple_effects(self, ticker: str, event: str = None) -> str:
        """Bir hisse için ripple effect analizi yap."""
        effects = []
        
        # Sektör ilişkilerini bul
        if ticker in self.SECTOR_RELATIONS:
            sectors = self.SECTOR_RELATIONS[ticker]
            effects.append(f"📊 {ticker} Sektörleri: {', '.join(sectors[:3])}")
        
        # İlgili ripple effects
        for event_name, impacts in self.RIPPLE_EFFECTS.items():
            if ticker in impacts:
                impact = impacts[ticker]
                direction = "📈" if impact > 0 else "📉"
                effects.append(f"{direction} {event_name}: {impact:+.0%} etki")
        
        if effects:
            return "\n<property_graph>\n🕸️ İLİŞKİ ANALİZİ:\n" + "\n".join(effects) + "\n</property_graph>\n"
        return ""


# ============================================
# I. MODEL ORCHESTRATOR (Multi-Model)
# ============================================
class ModelOrchestrator:
    """
    Sorgunun tipine göre en uygun modeli seçer.
    - Basit: Llama 3 (hızlı/ucuz)
    - Orta: Llama 3 70B veya Mixtral
    - Kritik: GPT-4o / Claude 3.5 (API)
    """
    
    MODELS = {
        "fast": {"name": "llama3", "provider": "ollama", "desc": "⚡ Hızlı"},
        "balanced": {"name": "llama3:70b", "provider": "ollama", "desc": "⚖️ Dengeli"},
        "premium": {"name": "gpt-4o-mini", "provider": "openai", "desc": "🏆 Premium"},
        "vision": {"name": "llama3.2-vision", "provider": "ollama", "desc": "👁️ Vision"},
    }
    
    def __init__(self, ollama_host: str, openai_key: str = None):
        self.ollama_host = ollama_host
        self.openai_key = openai_key or os.getenv("OPENAI_API_KEY")
        self.default_model = "fast"
    
    def select_model(self, query: str, complexity: QueryComplexity, has_image: bool = False) -> Dict:
        """Sorguya göre model seç."""
        if has_image:
            return self.MODELS["vision"]
        
        if complexity == QueryComplexity.SIMPLE:
            return self.MODELS["fast"]
        elif complexity == QueryComplexity.COMPLEX:
            # Premium model varsa kullan
            if self.openai_key:
                return self.MODELS["premium"]
            return self.MODELS["balanced"]
        
        return self.MODELS["fast"]
    
    def invoke(self, prompt: str, model_config: Dict) -> str:
        """Seçilen model ile çağrı yap."""
        if model_config["provider"] == "ollama":
            llm = OllamaLLM(model=model_config["name"], base_url=self.ollama_host)
            return llm.invoke(prompt)
        
        elif model_config["provider"] == "openai" and self.openai_key:
            try:
                import urllib.request
                import json
                
                payload = json.dumps({
                    "model": model_config["name"],
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2000
                }).encode('utf-8')
                
                req = urllib.request.Request(
                    "https://api.openai.com/v1/chat/completions",
                    data=payload,
                    headers={
                        'Content-Type': 'application/json',
                        'Authorization': f'Bearer {self.openai_key}'
                    }
                )
                
                with urllib.request.urlopen(req, timeout=30) as response:
                    result = json.loads(response.read().decode('utf-8'))
                    return result["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"{Fore.YELLOW}  → OpenAI hata, Ollama'ya geçiliyor: {e}{Style.RESET_ALL}", flush=True)
        
        # Fallback to Ollama
        llm = OllamaLLM(model="llama3", base_url=self.ollama_host)
        return llm.invoke(prompt)


# ============================================
# J. MARKET MONITOR (Event-Driven Architecture)
# ============================================
class MarketMonitor:
    """
    Olay güdümlü piyasa izleme servisi.
    RSI < 30, ani fiyat hareketleri veya volatilite artışında otomatik tetiklenir.
    """
    
    ALERT_THRESHOLDS = {
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "price_change_pct": 3.0,  # %3'ten fazla hareket
        "volatility_spike": 2.0,  # Normal volatilitenin 2 katı
    }
    
    def __init__(self, advisor_callback=None):
        self.advisor_callback = advisor_callback
        self.last_prices = {}
        self.last_rsi = {}
        self.alerts = []
        self.is_running = False
    
    def check_conditions(self, ticker: str, price: float, rsi: float, volatility: float = None) -> List[Dict]:
        """Piyasa koşullarını kontrol et ve alert oluştur."""
        alerts = []
        
        # RSI Oversold
        if rsi < self.ALERT_THRESHOLDS["rsi_oversold"]:
            alerts.append({
                "type": "RSI_OVERSOLD",
                "ticker": ticker,
                "value": rsi,
                "message": f"🔴 ACİL: {ticker} RSI {rsi:.1f} - Aşırı Satım Bölgesi!",
                "urgency": "HIGH"
            })
        
        # RSI Overbought
        elif rsi > self.ALERT_THRESHOLDS["rsi_overbought"]:
            alerts.append({
                "type": "RSI_OVERBOUGHT",
                "ticker": ticker,
                "value": rsi,
                "message": f"🟠 UYARI: {ticker} RSI {rsi:.1f} - Aşırı Alım Bölgesi!",
                "urgency": "MEDIUM"
            })
        
        # Price Change
        if ticker in self.last_prices:
            price_change = abs((price - self.last_prices[ticker]) / self.last_prices[ticker] * 100)
            if price_change > self.ALERT_THRESHOLDS["price_change_pct"]:
                direction = "📈" if price > self.last_prices[ticker] else "📉"
                alerts.append({
                    "type": "PRICE_SPIKE",
                    "ticker": ticker,
                    "value": price_change,
                    "message": f"{direction} ACİL: {ticker} %{price_change:.1f} hareket!",
                    "urgency": "HIGH"
                })
        
        # Volatility Spike
        if volatility and volatility > self.ALERT_THRESHOLDS["volatility_spike"]:
            alerts.append({
                "type": "VOLATILITY_SPIKE",
                "ticker": ticker,
                "value": volatility,
                "message": f"⚠️ {ticker} volatilite {volatility:.1f}x normal!",
                "urgency": "MEDIUM"
            })
        
        # Update last values
        self.last_prices[ticker] = price
        self.last_rsi[ticker] = rsi
        
        return alerts
    
    def trigger_analysis(self, alert: Dict) -> str:
        """Alert tetiklendiğinde analiz başlat."""
        if self.advisor_callback:
            print(f"\n{Fore.RED}{'='*60}{Style.RESET_ALL}", flush=True)
            print(f"{Fore.RED}🚨 MARKET ALERT: {alert['message']}{Style.RESET_ALL}", flush=True)
            print(f"{Fore.RED}{'='*60}{Style.RESET_ALL}", flush=True)
            
            return self.advisor_callback(
                ticker=alert["ticker"],
                tech_signals=f"ALERT: {alert['type']} - {alert['value']}",
                market_sentiment="Volatil"
            )
        return ""
    
    def generate_report(self, alerts: List[Dict]) -> str:
        """Alert raporu oluştur."""
        if not alerts:
            return ""
        
        report = "\n<market_alerts>\n🚨 PİYASA ALERT'LERİ:\n"
        for alert in alerts:
            urgency_emoji = "🔴" if alert["urgency"] == "HIGH" else "🟠"
            report += f"  {urgency_emoji} [{alert['type']}] {alert['message']}\n"
        report += "</market_alerts>\n"
        
        return report
    
    def start_monitoring(self, tickers: List[str], interval_seconds: int = 300):
        """Arka plan izleme başlat (threading ile)."""
        import threading
        import time
        
        def monitor_loop():
            self.is_running = True
            print(f"{Fore.GREEN}📡 MarketMonitor başlatıldı - {len(tickers)} sembol izleniyor{Style.RESET_ALL}", flush=True)
            
            while self.is_running:
                for ticker in tickers:
                    try:
                        if YFINANCE_AVAILABLE:
                            info = yf.Ticker(ticker).info
                            price = info.get("regularMarketPrice", 0)
                            # RSI için basit hesaplama (gerçek implementasyonda TA-Lib kullan)
                            rsi = 50  # Placeholder
                            
                            alerts = self.check_conditions(ticker, price, rsi)
                            for alert in alerts:
                                if alert["urgency"] == "HIGH":
                                    self.trigger_analysis(alert)
                    except:
                        pass
                
                time.sleep(interval_seconds)
        
        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()
        return thread
    
    def stop_monitoring(self):
        """İzlemeyi durdur."""
        self.is_running = False
        print(f"{Fore.YELLOW}📡 MarketMonitor durduruldu{Style.RESET_ALL}", flush=True)


# ============================================
# MAIN AI ADVISOR CLASS
# ============================================
class AIAdvisor:
    """
    AI Research Lab Level Trading Advisor.
    C-RAG + Python REPL + Cross-Encoder + SQLite Memory
    """
    
    def __init__(self, user_id: str = "default"):
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://ollama-service:11434")
        self.qdrant_url = os.getenv("QDRANT_URL", "http://qdrant-service:6333")
        self.user_id = user_id
        
        self.child_collection = "neural_trade_pro"
        self.parent_collection = "neural_trade_parent"
        
        # Core Components
        self.llm = OllamaLLM(model="llama3", base_url=self.ollama_host)
        self.dense_embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        self.client = QdrantClient(url=self.qdrant_url)
        self.bi_encoder = Ranker(model_name="ms-marco-MultiBERT-L-12")
        
        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.child_collection,
            embedding=self.dense_embeddings,
            sparse_embedding=self.sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        
        # AI Lab Components
        self.memory = SQLiteMemory()
        self.crag = CorrectiveRAG(self.llm)
        self.cross_encoder = CrossEncoderReranker()
        self.repl = PythonREPL()
        self.graph = KnowledgeGraph()
        self.rule_extractor = TradeRuleExtractor(self.llm)
        
        # Dynamic Intelligence Components
        self.agentic = AgenticToolUse(self.llm)
        self.vision = VisionRAG(self.ollama_host)
        
        # Hedge Fund AI Components
        self.property_graph = PropertyGraph(self.llm)
        self.orchestrator = ModelOrchestrator(self.ollama_host)

    def _batch_fetch_parents(self, docs: List[Document]) -> List[Document]:
        """Optimized parent fetch."""
        parent_ids = list(set(d.metadata.get("parent_id") for d in docs if d.metadata.get("parent_id")))
        if not parent_ids:
            return docs
        
        try:
            results = self.client.scroll(
                collection_name=self.parent_collection,
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(key="metadata.parent_id", match=models.MatchAny(any=parent_ids))
                ]),
                limit=min(len(parent_ids), 100),
                with_payload=True
            )
            
            if not results[0]:
                return docs
            
            parent_map = {p.payload.get("metadata", {}).get("parent_id"): Document(
                page_content=p.payload.get("page_content", ""),
                metadata=p.payload.get("metadata", {})
            ) for p in results[0]}
            
            expanded = []
            seen = set()
            for doc in docs:
                pid = doc.metadata.get("parent_id")
                if pid and pid in parent_map and pid not in seen:
                    expanded.append(parent_map[pid])
                    seen.add(pid)
            
            return expanded if expanded else docs
        except:
            return docs

    def _execute_calculation(self, query: str, tech_signals: str) -> str:
        """Python REPL ile finansal hesaplama yap."""
        # Hesaplama gerekiyor mu kontrol et
        calc_keywords = ["pip", "lot", "kaldıraç", "leverage", "pozisyon", "risk"]
        if not any(kw in query.lower() or kw in tech_signals.lower() for kw in calc_keywords):
            return ""
        
        print(f"{Fore.MAGENTA}  → Python REPL: Hesaplama yapılıyor...{Style.RESET_ALL}", flush=True)
        
        # Örnek hesaplamalar
        results = []
        
        # Risk/Reward hesapla (varsayılan değerlerle)
        results.append(self.repl.calculate_risk_reward(1.0850, 1.0800, 1.0950))
        results.append(self.repl.calculate_position_size(10000, 2, 50))
        
        if results:
            return f"\n🐍 PYTHON HESAPLAMALARI:\n  • " + "\n  • ".join(results) + "\n"
        return ""

    def _get_prompt(self, use_full: bool = True) -> str:
        if use_full:
            return """Sen AI Lab seviyesinde bir algoritmik trade danışmanısın.

<user_profile>
{personalization}
</user_profile>

<past_performance>
{past_performance}
</past_performance>

<graph_rag>
{graph_context}
</graph_rag>

<calculations>
{calculations}
</calculations>

<technical_data ticker="{ticker}">
{tech_signals}
Piyasa: {market_sentiment}
{fact_check}
</technical_data>

<strategies>
{context}
</strategies>

═══════════════════════════════════════════════════════════
🧠 CHAIN-OF-THOUGHT ANALİZ
═══════════════════════════════════════════════════════════

**ADIM 1 - TEKNİK ANALİZ:** Verileri analiz et.
**ADIM 2 - KİTAP STRATEJİLERİ:** Önerileri çıkar.
**ADIM 3 - KULLANICI PROFİLİ:** Risk toleransına göre ayarla.
**ADIM 4 - GEÇMİŞ PERFORMANS:** Önceki hatalardan ders al.
**ADIM 5 - EKONOMİK İLİŞKİLER:** Graph-RAG'dan çıkarımlar.
**ADIM 6 - HESAPLAMALAR:** Python REPL sonuçlarını değerlendir.
**ADIM 7 - RİSK DEĞERLENDİRMESİ:** 2+ risk belirt.
**ADIM 8 - NİHAİ KARAR:** 'AL 🟢', 'SAT 🔴' veya 'BEKLE 🟡'

ANALİZ:"""
        else:
            return """Trade danışmanısın.
<user_profile>{personalization}</user_profile>
<graph_rag>{graph_context}</graph_rag>
<calculations>{calculations}</calculations>
<technical_data ticker="{ticker}">{tech_signals}</technical_data>
Piyasa: {market_sentiment}
{fact_check}
<strategies>{context}</strategies>
AL/SAT/BEKLE öner:"""

    def analyze_trade(
        self, 
        ticker: str, 
        tech_signals: str, 
        market_sentiment: str = "Nötr"
    ):
        """
        AI Lab Level Trading Analizi.
        
        Pipeline:
        1. Semantic Router
        2. Adaptive RAG → Strateji seç
        3. Hybrid Search
        4. Bi-Encoder Reranking
        5. C-RAG (Alaka kontrolü)
        6. Cross-Encoder Reranking (ultra hassas)
        7. Parent Fetch + Summarize
        8. Python REPL (hesaplamalar)
        9. Graph-RAG
        10. Memory (kişiselleştirme)
        11. Chain-of-Thought LLM
        12. Guardrails
        """
        base_query = f"{ticker} için {tech_signals} verileri ile hangi strateji kullanılmalı?"
        
        # 1. Router
        if SemanticRouter.classify(base_query) == QueryType.GENERAL:
            return "Merhaba! 👋 Ben NeuralTrade AI Lab. Trading sorunuz var mı? 📈"
        
        # 2. Adaptive RAG
        complexity = AdaptiveRAG.assess(base_query)
        strategy = AdaptiveRAG.get_strategy(complexity)
        
        print(f"\n{Fore.YELLOW}[AI] {strategy['desc']} - {complexity.value.upper()}{Style.RESET_ALL}", flush=True)
        print(f"{Fore.CYAN}[AI] {ticker} için AI LAB analiz başlatılıyor...{Style.RESET_ALL}", flush=True)
        
        # 3. Hybrid Search
        docs = self.vectorstore.as_retriever(search_kwargs={"k": strategy["top_k"]}).invoke(base_query)
        print(f"{Fore.GREEN}  → {len(docs)} döküman bulundu{Style.RESET_ALL}", flush=True)
        
        # 4. Bi-Encoder Reranking
        if len(docs) > 1:
            passages = [{"id": i, "text": d.page_content} for i, d in enumerate(docs)]
            rerank = self.bi_encoder.rerank(RerankRequest(query=base_query, passages=passages))
            docs = [docs[r["id"]] for r in rerank[:7]]
            print(f"{Fore.GREEN}  → Bi-Encoder: {len(docs)} döküman{Style.RESET_ALL}", flush=True)
        
        # 5. C-RAG (Alaka Kontrolü)
        web_news = ""
        if strategy["use_crag"]:
            print(f"{Fore.MAGENTA}  → C-RAG: Alaka kontrolü...{Style.RESET_ALL}", flush=True)
            original_count = len(docs)
            docs = self.crag.filter_documents(base_query, docs)
            print(f"{Fore.GREEN}  → C-RAG: {len(docs)} alakalı döküman{Style.RESET_ALL}", flush=True)
            
            # Web Fallback: Dökümanlar alakasız ise haberleri çek
            if len(docs) < original_count * 0.3:  # %70'ten fazlası filtrelendiyse
                web_news = self.crag.web_search_fallback(ticker, base_query)
        
        # 6. Cross-Encoder Reranking (son 5 için)
        if strategy["use_cross"] and self.cross_encoder.available:
            print(f"{Fore.MAGENTA}  → Cross-Encoder: Ultra hassas sıralama...{Style.RESET_ALL}", flush=True)
            docs = self.cross_encoder.rerank(base_query, docs, top_k=3)
            print(f"{Fore.GREEN}  → Cross-Encoder: {len(docs)} döküman (ultra hassas){Style.RESET_ALL}", flush=True)
        else:
            docs = docs[:5]
        
        # 7. Parent Fetch
        docs = self._batch_fetch_parents(docs)
        
        # Context oluştur
        context = "\n\n---\n\n".join([f"[{d.metadata.get('category', '?')}] {d.page_content[:800]}" for d in docs[:3]])
        
        # 8. Trade Rule Extraction (Entry/Exit Seviyeleri)
        trade_rules = ""
        current_price = 0.0
        current_rsi = 40.0
        
        # Fiyat ve RSI'ı tech_signals'dan çıkar
        import re
        price_match = re.search(r'Fiyat:\s*([\d.]+)', tech_signals)
        rsi_match = re.search(r'RSI.*?:\s*([\d.]+)', tech_signals)
        if price_match:
            current_price = float(price_match.group(1))
        if rsi_match:
            current_rsi = float(rsi_match.group(1))
        
        if current_price > 0:
            # Dökümanlardan trade kuralları çıkar
            trade_rules = self.rule_extractor.extract_trade_rules(docs, ticker, current_price)
            # Otomatik seviye hesaplama ekle
            trade_rules += self.rule_extractor.calculate_levels(current_price, current_rsi)
            print(f"{Fore.GREEN}  → Trade Rules: Entry/Exit seviyeleri hesaplandı{Style.RESET_ALL}", flush=True)
        
        # 9. Python REPL
        calculations = self._execute_calculation(base_query, tech_signals)
        
        # 10. Graph-RAG
        graph_context = self.graph.get_graph_context(base_query)
        if graph_context:
            print(f"{Fore.MAGENTA}  → Graph-RAG: Ekonomik ilişkiler bulundu{Style.RESET_ALL}", flush=True)
        
        # 11. Memory
        personalization = self.memory.get_personalization_context(self.user_id)
        print(f"{Fore.CYAN}  → SQLite Memory: Profil yüklendi{Style.RESET_ALL}", flush=True)
        
        # 12. Fact-Check
        fact_check = ""
        if YFINANCE_AVAILABLE and ticker not in ["GENEL", "EURUSD"]:
            try:
                info = yf.Ticker(ticker).info
                price = info.get("regularMarketPrice", info.get("currentPrice"))
                if price:
                    change = ((price - info.get("previousClose", price)) / info.get("previousClose", price) * 100)
                    fact_check = f"\n✅ CANLI: {ticker} ${price:.2f} ({change:+.2f}%)\n"
                    print(f"{Fore.GREEN}  → Fact-Check: ${price:.2f}{Style.RESET_ALL}", flush=True)
            except:
                pass
        
        # Web haberlerini ekle
        if web_news:
            context = web_news + "\n" + context
        
        # Trade kurallarını ekle
        if trade_rules:
            context = trade_rules + "\n" + context
        
        # 13. Backtest Learning - Geçmiş performans analizi
        past_performance = self.memory.analyze_past_performance(self.user_id)
        if past_performance:
            print(f"{Fore.CYAN}  → Backtest Learning: Geçmiş dersler yüklendi{Style.RESET_ALL}", flush=True)
        
        # 14. Agentic Tool Planning (dinamik)
        if complexity == QueryComplexity.COMPLEX:
            planned_tools = self.agentic.plan_tools(base_query)
            print(f"{Fore.MAGENTA}  → Agentic: Planlanan araçlar: {planned_tools}{Style.RESET_ALL}", flush=True)
        
        # 15. LLM Generation
        template = self._get_prompt(complexity == QueryComplexity.COMPLEX)
        prompt = ChatPromptTemplate.from_template(template)
        
        chain = ({
            "context": lambda x: context,
            "ticker": lambda x: ticker,
            "tech_signals": lambda x: tech_signals,
            "market_sentiment": lambda x: market_sentiment,
            "fact_check": lambda x: fact_check,
            "personalization": lambda x: personalization,
            "graph_context": lambda x: graph_context,
            "calculations": lambda x: calculations,
            "past_performance": lambda x: past_performance
        } | prompt | self.llm | StrOutputParser())
        
        try:
            print(f"{Fore.YELLOW}  → LLM: AI Lab analiz...{Style.RESET_ALL}", flush=True)
            response = chain.invoke(base_query)
            
            # Extract action and save
            action = "BEKLE"
            if "AL" in response.upper():
                action = "AL"
            elif "SAT" in response.upper():
                action = "SAT"
            
            self.memory.add_recommendation(self.user_id, ticker, action, 85)
            
            print(f"{Fore.GREEN}✅ AI Lab analiz tamamlandı{Style.RESET_ALL}", flush=True)
            return response
        except Exception as e:
            return f"❌ Hata: {str(e)}"


# ============================================
# HELPER FUNCTIONS
# ============================================
def set_user_preference(advisor: AIAdvisor, key: str, value: str):
    """Kullanıcı tercihini güncelle."""
    advisor.memory.update_preference(advisor.user_id, key, value)
    print(f"{Fore.GREEN}✅ Tercih güncellendi: {key} = {value}{Style.RESET_ALL}")


# ============================================
# TEST
# ============================================
if __name__ == "__main__":
    print(f"\n{Fore.YELLOW}{'='*60}")
    print("🔬 NEURALTRADE AI LAB TEST")
    print(f"{'='*60}{Style.RESET_ALL}\n")
    
    advisor = AIAdvisor(user_id="test_user")
    
    # Tercih ayarla
    set_user_preference(advisor, "risk_tolerance", "moderate")
    set_user_preference(advisor, "trading_style", "swing")
    
    sample_signals = """
    💰 Fiyat: 267.26
    📈 RSI(14): 40.41
    📊 Trend: DÜŞÜŞ
    🔲 FVG: 12 adet
    """
    
    result = advisor.analyze_trade(
        ticker="AAPL",
        tech_signals=sample_signals,
        market_sentiment="Faiz artışı ve enflasyon endişeleri"
    )
    
    print(f"\n{Fore.GREEN}{'='*60}")
    print("📋 AI RAPOR (AI LAB LEVEL)")
    print(f"{'='*60}{Style.RESET_ALL}\n")
    print(result)