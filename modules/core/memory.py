"""
SQLite Memory - Persistent User Memory with Backtest Learning
Author: Erdinc Erdogan
Purpose: Provides thread-safe SQLite-based user memory for storing conversation history,
trading decisions, and backtest results for learning from past performance.
References:
- SQLite Thread-Safety Patterns
- Backtest Learning and Strategy Adaptation
- Persistent Storage for AI Agents
Usage:
    memory = SQLiteMemory(db_path="./data/user_memory.db")
    memory.store_interaction(user_id="001", message="...", response="...")
    history = memory.get_history(user_id="001", limit=10)
"""
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict


class SQLiteMemory:
    """
    SQLite tabanlı kullanıcı hafızası.
    Thread-safe, backtest learning destekli.
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
        
        # Backtest Results tablosu
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
        self.get_user_profile(user_id)
        
        cursor.execute(f'''
            UPDATE users SET {key} = ?, last_active = ? WHERE user_id = ?
        ''', (value, datetime.now().isoformat(), user_id))
        
        conn.commit()
        conn.close()
    
    def add_recommendation(self, user_id: str, ticker: str, action: str, confidence: int, entry_price: float = 0):
        """Öneriyi kaydet."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO recommendations (user_id, ticker, action, entry_price, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, ticker, action, entry_price, confidence, datetime.now().isoformat()))
        
        cursor.execute('''
            UPDATE users SET total_trades = total_trades + 1, last_active = ?
            WHERE user_id = ?
        ''', (datetime.now().isoformat(), user_id))
        
        conn.commit()
        rec_id = cursor.lastrowid
        conn.close()
        return rec_id
    
    def record_trade_result(self, recommendation_id: int, exit_price: float, entry_price: float, outcome: str, lesson: str):
        """Trade sonucunu kaydet (Backtest Learning)."""
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
        """Geçmiş performansı analiz et."""
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
        if lessons:
            context += "• Son Dersler:\n"
            for l in lessons[:3]:
                if l["lesson"]:
                    context += f"  - {l['ticker']}: {l['lesson']}\n"
        
        context += "</backtest_learning>\n"
        return context
    
    def get_personalization_context(self, user_id: str) -> str:
        """Kişiselleştirme bağlamı oluştur."""
        profile = self.get_user_profile(user_id)
        
        return f"""
<user_profile>
👤 KULLANICI PROFİLİ:
• Risk Toleransı: {profile['risk_tolerance'].upper()}
• Trading Stili: {profile['trading_style'].upper()}
• Kaldıraç: {profile['leverage_pref'].upper()}
• Toplam Analiz: {profile['total_trades']}
</user_profile>
"""
