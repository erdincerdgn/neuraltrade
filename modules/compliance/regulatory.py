"""
Regulatory Compliance Engine - Multi-Asset Compliance Framework
Author: Erdinc Erdogan
Purpose: Implements regulatory compliance checks including wash trading detection, spoofing
prevention, and KYT security modules for stocks, forex, bonds, and crypto.
References:
- SEC/FINRA Market Manipulation Rules
- MiFID II Transaction Reporting
- Anti-Money Laundering (AML) Requirements
Usage:
    detector = WashTradingDetector()
    is_wash = detector.check(agent_orders, window_seconds=60)
"""
import os
import hashlib
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from collections import defaultdict
from colorama import Fore, Style


class WashTradingDetector:
    """
    Wash Trading Detection.
    
    Botun veya kullanıcının kendi kendine alım-satım yapmasını tespit eder.
    Bu, piyasa manipülasyonu suçudur ve engellenmelidir.
    
    Kontroller:
    - Aynı zamanda zıt emirler
    - Circular trading patterns
    - Matched trades at same price
    """
    
    def __init__(self, time_window_seconds: int = 60):
        """
        Args:
            time_window_seconds: Wash trading tespit penceresi
        """
        self.time_window = timedelta(seconds=time_window_seconds)
        self.order_history = []
        self.trade_history = []
        self.violations = []
        self.account_id = os.getenv("ACCOUNT_ID", "DEFAULT")
    
    def check_order(self, order: Dict) -> Dict:
        """
        Emri wash trading için kontrol et.
        
        Args:
            order: {side, symbol, size, price, timestamp, account_id}
        """
        now = datetime.now()
        order["timestamp"] = order.get("timestamp", now)
        order["account_id"] = order.get("account_id", self.account_id)
        
        violations = []
        
        # Kontrol 1: Aynı sembolde zıt emir var mı?
        opposite_side = "SELL" if order["side"] == "BUY" else "BUY"
        recent_opposite = self._find_recent_orders(
            order["symbol"], opposite_side, order["timestamp"]
        )
        
        if recent_opposite:
            for opp in recent_opposite:
                # Aynı fiyat veya çok yakın?
                price_diff = abs(order.get("price", 0) - opp.get("price", 0))
                mid_price = (order.get("price", 1) + opp.get("price", 1)) / 2
                
                if mid_price > 0 and price_diff / mid_price < 0.001:  # %0.1'den az fark
                    violations.append({
                        "type": "MATCHED_OPPOSITE_ORDER",
                        "severity": "HIGH",
                        "original_order": opp,
                        "new_order": order,
                        "reason": "Aynı fiyatta zıt emir tespit edildi"
                    })
        
        # Kontrol 2: Aynı hesaptan circular trade
        if order.get("account_id") == self.account_id:
            circular = self._detect_circular_pattern(order)
            if circular:
                violations.append({
                    "type": "CIRCULAR_WASH",
                    "severity": "CRITICAL",
                    "pattern": circular,
                    "reason": "Circular trading pattern tespit edildi"
                })
        
        # Sonuç
        is_wash = len(violations) > 0
        
        if is_wash:
            self.violations.extend(violations)
            print(f"{Fore.RED}🚨 WASH TRADING TESPİT: {order['symbol']}{Style.RESET_ALL}", flush=True)
        
        # Order history'e ekle
        self.order_history.append(order)
        
        return {
            "order": order,
            "is_wash_trading": is_wash,
            "violations": violations,
            "action": "BLOCK" if is_wash else "ALLOW"
        }
    
    def _find_recent_orders(self, symbol: str, side: str, timestamp: datetime) -> List[Dict]:
        """Son emirleri bul."""
        recent = []
        cutoff = timestamp - self.time_window
        
        for order in self.order_history:
            if (order["symbol"] == symbol and 
                order["side"] == side and
                order.get("timestamp", datetime.min) > cutoff):
                recent.append(order)
        
        return recent
    
    def _detect_circular_pattern(self, new_order: Dict) -> Optional[Dict]:
        """Circular pattern tespit."""
        symbol = new_order["symbol"]
        
        recent = [o for o in self.order_history[-20:] if o["symbol"] == symbol]
        
        if len(recent) < 4:
            return None
        
        # Buy-Sell-Buy-Sell pattern
        sides = [o["side"] for o in recent[-4:]]
        if sides == ["BUY", "SELL", "BUY", "SELL"] or sides == ["SELL", "BUY", "SELL", "BUY"]:
            return {
                "pattern": "ALTERNATING",
                "count": 4,
                "orders": recent[-4:]
            }
        
        return None
    
    def get_violation_report(self) -> str:
        """İhlal raporu."""
        return f"""
<wash_trading_detector>
🚨 WASH TRADING DENETİMİ
════════════════════════════════════════

📊 İSTATİSTİK:
  • Toplam Emir: {len(self.order_history)}
  • İhlal: {len(self.violations)}
  
⚠️ SON İHLALLER:
""" + "\n".join([f"  • {v['type']}: {v['reason']}" for v in self.violations[-5:]]) + "\n</wash_trading_detector>"


class SpoofingGuard:
    """
    Spoofing Guard.
    
    Botun piyasayı manipüle edip etmediğini kontrol eder:
    - Büyük emirler verip hemen iptal etme
    - Fiyatı yapay olarak hareket ettirme
    - Layering (katmanlı emirler)
    """
    
    def __init__(self,
                 cancel_threshold: float = 0.9,  # %90+ iptal = şüpheli
                 time_window_seconds: int = 30):
        """
        Args:
            cancel_threshold: İptal oranı eşiği
            time_window_seconds: Analiz penceresi
        """
        self.cancel_threshold = cancel_threshold
        self.time_window = timedelta(seconds=time_window_seconds)
        self.order_events = []  # {event: SUBMIT/CANCEL/FILL, order_id, timestamp}
        self.alerts = []
    
    def log_order_event(self, event_type: str, order: Dict) -> Dict:
        """
        Emir olayı kaydet.
        
        Args:
            event_type: SUBMIT / CANCEL / FILL
            order: Emir detayları
        """
        event = {
            "event": event_type,
            "order_id": order.get("id"),
            "symbol": order.get("symbol"),
            "side": order.get("side"),
            "size": order.get("size"),
            "price": order.get("price"),
            "timestamp": datetime.now()
        }
        
        self.order_events.append(event)
        
        # Analiz
        check_result = self._analyze_recent_behavior()
        
        return {
            "event_logged": event,
            "analysis": check_result
        }
    
    def _analyze_recent_behavior(self) -> Dict:
        """Son davranışları analiz et."""
        now = datetime.now()
        cutoff = now - self.time_window
        
        recent = [e for e in self.order_events if e["timestamp"] > cutoff]
        
        if len(recent) < 5:
            return {"status": "INSUFFICIENT_DATA"}
        
        # İptal oranı
        submits = sum(1 for e in recent if e["event"] == "SUBMIT")
        cancels = sum(1 for e in recent if e["event"] == "CANCEL")
        fills = sum(1 for e in recent if e["event"] == "FILL")
        
        cancel_ratio = cancels / submits if submits > 0 else 0
        
        alerts = []
        
        # Yüksek iptal oranı
        if cancel_ratio > self.cancel_threshold:
            alerts.append({
                "type": "HIGH_CANCEL_RATIO",
                "severity": "HIGH",
                "cancel_ratio": cancel_ratio,
                "message": f"İptal oranı çok yüksek: %{cancel_ratio*100:.0f}"
            })
        
        # Layering tespiti
        layering = self._detect_layering(recent)
        if layering:
            alerts.append({
                "type": "LAYERING_DETECTED",
                "severity": "CRITICAL",
                "details": layering,
                "message": "Layering manipülasyonu tespit edildi"
            })
        
        if alerts:
            self.alerts.extend(alerts)
            print(f"{Fore.RED}⚔️ SPOOFING ALARM: {alerts[0]['type']}{Style.RESET_ALL}", flush=True)
        
        return {
            "status": "ALERT" if alerts else "OK",
            "cancel_ratio": cancel_ratio,
            "submits": submits,
            "cancels": cancels,
            "fills": fills,
            "alerts": alerts
        }
    
    def _detect_layering(self, events: List[Dict]) -> Optional[Dict]:
        """Layering tespiti."""
        # Aynı sembolde birden fazla seviyede emir
        symbol_orders = defaultdict(list)
        
        for e in events:
            if e["event"] == "SUBMIT":
                symbol_orders[e["symbol"]].append(e)
        
        for symbol, orders in symbol_orders.items():
            if len(orders) >= 3:
                prices = [o["price"] for o in orders if o["price"]]
                if len(set(prices)) >= 3:  # 3+ farklı fiyat
                    # Hepsi aynı tarafta mı?
                    sides = [o["side"] for o in orders]
                    if len(set(sides)) == 1:
                        return {
                            "symbol": symbol,
                            "num_layers": len(prices),
                            "side": sides[0],
                            "prices": prices
                        }
        
        return None
    
    def pre_order_check(self, order: Dict) -> Dict:
        """
        Emir göndermeden önce kontrol.
        
        Potansiyel spoofing engellemesi.
        """
        # Çok büyük emir + anlık iptal geçmişi
        recent_cancels = sum(
            1 for e in self.order_events[-20:]
            if e["event"] == "CANCEL" and e["symbol"] == order.get("symbol")
        )
        
        if recent_cancels > 10 and order.get("size", 0) > 10000:
            return {
                "allow": False,
                "reason": "Yüksek iptal geçmişi + büyük emir = potansiyel spoofing",
                "suggestion": "Daha küçük emirler gönderin"
            }
        
        return {"allow": True}


class KYTAnalyzer:
    """
    KYT (Know Your Transaction) Analyzer.
    
    İşlem yapılacak havuzun/token'ın kara para, 
    terörizm finansmanı vb. ile ilişkili olup olmadığını kontrol eder.
    
    Kripto için: Wallet/Contract risk analizi
    Fiat için: Karşı taraf risk kontrolü
    """
    
    # Bilinen riskli adresler (örnek)
    BLACKLIST = {
        "0x1234...": "Sanctions - OFAC",
        "0x5678...": "Hack - Exchange exploit",
        "0xabcd...": "Mixer - Tornado Cash related",
    }
    
    # Risk kategorileri
    RISK_CATEGORIES = {
        "sanctions": {"weight": 1.0, "description": "Yaptırım listesinde"},
        "mixer": {"weight": 0.8, "description": "Mixer/Tumbler ilişkili"},
        "darknet": {"weight": 0.9, "description": "Darknet market ilişkili"},
        "scam": {"weight": 0.7, "description": "Bilinen scam"},
        "hack": {"weight": 0.9, "description": "Hack/exploit ilişkili"},
        "high_risk_exchange": {"weight": 0.5, "description": "Yüksek riskli borsa"},
    }
    
    def __init__(self):
        self.checked_addresses = {}
        self.blocked_transactions = []
    
    def analyze_address(self, address: str, asset_type: str = "crypto") -> Dict:
        """
        Adres/hesap risk analizi.
        
        Args:
            address: Wallet adresi veya hesap numarası
            asset_type: crypto / fiat
        """
        print(f"{Fore.CYAN}🔍 KYT Analizi: {address[:10]}...{Style.RESET_ALL}", flush=True)
        
        risk_score = 0
        risk_flags = []
        
        # Kara liste kontrolü
        if address in self.BLACKLIST:
            return {
                "address": address,
                "risk_score": 1.0,
                "risk_level": "CRITICAL",
                "flags": [{"type": "BLACKLIST", "reason": self.BLACKLIST[address]}],
                "recommendation": "BLOCK",
                "allow_transaction": False
            }
        
        # Simüle edilmiş risk analizi
        # Gerçekte: Chainalysis, Elliptic, TRM Labs API
        
        # Adres pattern analizi
        if address.startswith("0x"):  # Ethereum-like
            risk_score += self._analyze_eth_address(address)
        
        # Mixer ilişki kontrolü (simüle)
        if hash(address) % 100 < 5:  # %5 şans
            risk_score += 0.8
            risk_flags.append({
                "type": "mixer",
                "confidence": 0.7,
                "reason": "Potansiyel mixer ilişkisi tespit edildi"
            })
        
        # Risk seviyesi
        if risk_score >= 0.8:
            level = "CRITICAL"
            allow = False
        elif risk_score >= 0.5:
            level = "HIGH"
            allow = False
        elif risk_score >= 0.3:
            level = "MEDIUM"
            allow = True  # Uyarı ile
        else:
            level = "LOW"
            allow = True
        
        result = {
            "address": address,
            "asset_type": asset_type,
            "risk_score": risk_score,
            "risk_level": level,
            "flags": risk_flags,
            "recommendation": "BLOCK" if not allow else "PROCEED" if level == "LOW" else "PROCEED_WITH_CAUTION",
            "allow_transaction": allow,
            "timestamp": datetime.now().isoformat()
        }
        
        self.checked_addresses[address] = result
        
        if not allow:
            self.blocked_transactions.append(result)
            print(f"{Fore.RED}🚫 BLOCKED: Risk={level}{Style.RESET_ALL}", flush=True)
        
        return result
    
    def _analyze_eth_address(self, address: str) -> float:
        """Ethereum adres analizi."""
        risk = 0.0
        
        # Yeni adres (az işlem) = potansiyel risk
        address_hash = int(hashlib.md5(address.encode()).hexdigest(), 16)
        
        if address_hash % 1000 < 10:  # %1 -> yeni adres simülasyonu
            risk += 0.2
        
        return risk
    
    def check_transaction(self, 
                         from_address: str,
                         to_address: str,
                         amount: float,
                         asset: str) -> Dict:
        """
        İşlem kontrol et.
        """
        from_analysis = self.analyze_address(from_address)
        to_analysis = self.analyze_address(to_address)
        
        # En yüksek riski al
        max_risk = max(from_analysis["risk_score"], to_analysis["risk_score"])
        
        allow = from_analysis["allow_transaction"] and to_analysis["allow_transaction"]
        
        return {
            "from": from_address,
            "to": to_address,
            "amount": amount,
            "asset": asset,
            "risk_score": max_risk,
            "allow": allow,
            "from_analysis": from_analysis,
            "to_analysis": to_analysis
        }
    
    def generate_compliance_report(self) -> str:
        """Uyumluluk raporu."""
        report = f"""
<kyt_compliance>
🔐 KYT UYUMLULUK RAPORU
════════════════════════════════════════

📊 İSTATİSTİK:
  • Kontrol Edilen Adres: {len(self.checked_addresses)}
  • Engellenen İşlem: {len(self.blocked_transactions)}

📋 RİSK DAĞILIMI:
"""
        risk_counts = defaultdict(int)
        for addr, analysis in self.checked_addresses.items():
            risk_counts[analysis["risk_level"]] += 1
        
        for level, count in risk_counts.items():
            report += f"  • {level}: {count}\n"
        
        report += """
⚠️ SON ENGELLEMELer:
"""
        for tx in self.blocked_transactions[-5:]:
            report += f"  • {tx['address'][:15]}... - {tx['risk_level']}\n"
        
        report += "</kyt_compliance>\n"
        return report
