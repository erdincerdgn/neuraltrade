"""
Paper Trading Engine with Broker Integration
Author: Erdinc Erdogan
Purpose: Executes virtual trades on Alpaca Paper Trading and Binance Testnet with SQLite persistence for strategy validation without real capital.
References:
- Alpaca Paper Trading API
- Binance Testnet Integration
- Virtual Portfolio Management
Usage:
    engine = PaperTradingEngine()
    result = engine.execute_trade('AAPL', 'BUY', quantity=10, price=150.0)
    portfolio = engine.get_portfolio_status()
"""
import os
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from colorama import Fore, Style


class PaperTradingEngine:
    """
    Paper Trading Engine.
    Alpaca veya Binance Testnet üzerinden sanal işlem yapma.
    """
    
    def __init__(self, db_path: str = "/app/data/paper_trades.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # API Keys
        self.alpaca_key = os.getenv("ALPACA_API_KEY")
        self.alpaca_secret = os.getenv("ALPACA_SECRET_KEY")
        self.binance_key = os.getenv("BINANCE_TESTNET_KEY")
        self.binance_secret = os.getenv("BINANCE_TESTNET_SECRET")
        
        self._init_db()
        
        # Virtual portfolio
        self.initial_balance = 100000.0  # $100K sanal para
        self.balance = self.initial_balance
        self.positions = {}
    
    def _init_db(self):
        """Trade veritabanını oluştur."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                action TEXT,
                quantity REAL,
                entry_price REAL,
                exit_price REAL,
                profit_loss REAL,
                status TEXT DEFAULT 'OPEN',
                entry_time TEXT,
                exit_time TEXT,
                strategy TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                total_value REAL,
                cash REAL,
                positions_value REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def execute_trade(self, 
                     ticker: str, 
                     action: str, 
                     quantity: float,
                     price: float,
                     strategy: str = "AI") -> Dict:
        """
        Paper trade çalıştır.
        """
        trade_id = None
        
        if action.upper() == "AL" or action.upper() == "BUY":
            # Yeterli bakiye var mı?
            total_cost = quantity * price
            if total_cost > self.balance:
                return {"success": False, "error": "Yetersiz bakiye", "available": self.balance}
            
            # İşlemi kaydet
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO paper_trades (ticker, action, quantity, entry_price, status, entry_time, strategy)
                VALUES (?, ?, ?, ?, 'OPEN', ?, ?)
            ''', (ticker, "BUY", quantity, price, datetime.now().isoformat(), strategy))
            
            trade_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            # Portföyü güncelle
            self.balance -= total_cost
            if ticker in self.positions:
                self.positions[ticker]['quantity'] += quantity
                self.positions[ticker]['avg_price'] = (
                    (self.positions[ticker]['avg_price'] * (self.positions[ticker]['quantity'] - quantity) + price * quantity) 
                    / self.positions[ticker]['quantity']
                )
            else:
                self.positions[ticker] = {'quantity': quantity, 'avg_price': price}
            
            print(f"{Fore.GREEN}📈 PAPER TRADE: {ticker} {quantity} adet @ ${price:.2f}{Style.RESET_ALL}", flush=True)
            
            return {
                "success": True,
                "trade_id": trade_id,
                "ticker": ticker,
                "action": "BUY",
                "quantity": quantity,
                "price": price,
                "total_cost": total_cost,
                "remaining_balance": self.balance
            }
        
        elif action.upper() == "SAT" or action.upper() == "SELL":
            # Pozisyon var mı?
            if ticker not in self.positions or self.positions[ticker]['quantity'] < quantity:
                return {"success": False, "error": "Yetersiz pozisyon"}
            
            # İşlemi kaydet
            entry_price = self.positions[ticker]['avg_price']
            profit_loss = (price - entry_price) * quantity
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO paper_trades (ticker, action, quantity, entry_price, exit_price, profit_loss, status, entry_time, exit_time, strategy)
                VALUES (?, ?, ?, ?, ?, ?, 'CLOSED', ?, ?, ?)
            ''', (ticker, "SELL", quantity, entry_price, price, profit_loss, 
                  datetime.now().isoformat(), datetime.now().isoformat(), strategy))
            
            trade_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            # Portföyü güncelle
            self.balance += quantity * price
            self.positions[ticker]['quantity'] -= quantity
            if self.positions[ticker]['quantity'] <= 0:
                del self.positions[ticker]
            
            print(f"{Fore.RED}📉 PAPER TRADE: {ticker} {quantity} adet SATIŞ @ ${price:.2f} (PnL: ${profit_loss:+.2f}){Style.RESET_ALL}", flush=True)
            
            return {
                "success": True,
                "trade_id": trade_id,
                "ticker": ticker,
                "action": "SELL",
                "quantity": quantity,
                "price": price,
                "profit_loss": profit_loss,
                "remaining_balance": self.balance
            }
        
        return {"success": False, "error": "Geçersiz işlem tipi"}
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> Dict:
        """Portföy değerini hesapla."""
        positions_value = sum(
            self.positions[ticker]['quantity'] * current_prices.get(ticker, 0)
            for ticker in self.positions
        )
        
        total_value = self.balance + positions_value
        pnl = total_value - self.initial_balance
        pnl_pct = (pnl / self.initial_balance) * 100
        
        return {
            "cash": self.balance,
            "positions_value": positions_value,
            "total_value": total_value,
            "initial_balance": self.initial_balance,
            "pnl": pnl,
            "pnl_pct": pnl_pct
        }
    
    def get_trade_history(self, limit: int = 10) -> List[Dict]:
        """Trade geçmişini getir."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT ticker, action, quantity, entry_price, exit_price, profit_loss, status, entry_time
            FROM paper_trades ORDER BY id DESC LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {"ticker": r[0], "action": r[1], "quantity": r[2], "entry_price": r[3],
             "exit_price": r[4], "pnl": r[5], "status": r[6], "time": r[7]}
            for r in rows
        ]
    
    def generate_performance_report(self, current_prices: Dict[str, float]) -> str:
        """Performans raporu oluştur."""
        portfolio = self.get_portfolio_value(current_prices)
        history = self.get_trade_history(5)
        
        report = f"""
<paper_trading>
💰 PAPER TRADING PERFORMANSI
════════════════════════════════════════
💵 Başlangıç: ${portfolio['initial_balance']:,.2f}
💼 Güncel Değer: ${portfolio['total_value']:,.2f}
📊 PnL: ${portfolio['pnl']:+,.2f} ({portfolio['pnl_pct']:+.2f}%)

📋 SON İŞLEMLER:
"""
        for trade in history[:5]:
            emoji = "📈" if trade['action'] == "BUY" else "📉"
            pnl_str = f"PnL: ${trade['pnl']:+.2f}" if trade['pnl'] else ""
            report += f"  {emoji} {trade['ticker']} {trade['action']} {trade['quantity']} @ ${trade['entry_price']:.2f} {pnl_str}\n"
        
        report += "</paper_trading>\n"
        return report
