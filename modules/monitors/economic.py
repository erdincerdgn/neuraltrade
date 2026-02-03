
"""
Economic Calendar Monitor
Author: Erdinc Erdogan
Purpose: Tracks high-impact economic events (FOMC, NFP, CPI) and provides volatility guard periods to avoid trading during major announcements.
References:
- Economic Event Impact Analysis
- ForexFactory Calendar Integration
- Volatility Guard Mechanisms
Usage:
    calendar = EconomicCalendar()
    should_avoid, reason = calendar.should_avoid_trading("AAPL")
    events = calendar.fetch_economic_calendar(days_ahead=7)
"""

import os
import re
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from colorama import Fore, Style


class EconomicCalendar:
    """
    Ekonomik Takvim İzleyici.
    Yüksek etkili olayları takip eder ve volatility guard sağlar.
    """
    
    # Yüksek etkili olaylar ve etkileri
    HIGH_IMPACT_EVENTS = {
        "FOMC": {"impact": "HIGH", "assets": ["USD", "SPY", "QQQ", "DXY"], "direction": "variable"},
        "NFP": {"impact": "HIGH", "assets": ["USD", "SPY"], "direction": "variable"},
        "CPI": {"impact": "HIGH", "assets": ["USD", "XAU", "TLT"], "direction": "inverse"},
        "PPI": {"impact": "MEDIUM", "assets": ["USD"], "direction": "inverse"},
        "GDP": {"impact": "HIGH", "assets": ["USD", "SPY"], "direction": "direct"},
        "UNEMPLOYMENT": {"impact": "MEDIUM", "assets": ["USD"], "direction": "inverse"},
        "RETAIL_SALES": {"impact": "MEDIUM", "assets": ["XRT", "AMZN"], "direction": "direct"},
        "FED_SPEECH": {"impact": "MEDIUM", "assets": ["USD", "SPY"], "direction": "variable"},
        "ECB_RATE": {"impact": "HIGH", "assets": ["EUR", "EWQ"], "direction": "direct"},
        "BOJ_RATE": {"impact": "HIGH", "assets": ["JPY", "EWJ"], "direction": "direct"},
    }
    
    # Volatilite guard süresi (saat)
    VOLATILITY_GUARD_HOURS = {
        "HIGH": 2,
        "MEDIUM": 1,
        "LOW": 0.5
    }
    
    def __init__(self):
        self.cached_events = []
        self.last_fetch = None
        self.fmp_api_key = os.getenv("FMP_API_KEY")
    
    def fetch_economic_calendar(self, days_ahead: int = 7) -> List[Dict]:
        """
        Ekonomik takvimi çek (ForexFactory/Investing.com veya FMP API).
        """
        events = []
        
        # Örnek veriler (gerçek API olmadan demo)
        # Gerçek implementasyonda: ForexFactory scraping veya FMP API
        sample_events = [
            {
                "name": "Fed Interest Rate Decision",
                "code": "FOMC",
                "date": (datetime.now() + timedelta(days=3)).isoformat(),
                "impact": "HIGH",
                "previous": "5.25%",
                "forecast": "5.25%",
                "actual": None
            },
            {
                "name": "Nonfarm Payrolls",
                "code": "NFP",
                "date": (datetime.now() + timedelta(days=5)).isoformat(),
                "impact": "HIGH",
                "previous": "216K",
                "forecast": "200K",
                "actual": None
            },
            {
                "name": "Consumer Price Index",
                "code": "CPI",
                "date": (datetime.now() + timedelta(days=10)).isoformat(),
                "impact": "HIGH",
                "previous": "3.2%",
                "forecast": "3.1%",
                "actual": None
            }
        ]
        
        # FMP API varsa gerçek veri çek
        if self.fmp_api_key:
            try:
                import urllib.request
                from_date = datetime.now().strftime("%Y-%m-%d")
                to_date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
                
                url = f"https://financialmodelingprep.com/api/v3/economic_calendar?from={from_date}&to={to_date}&apikey={self.fmp_api_key}"
                req = urllib.request.Request(url)
                
                with urllib.request.urlopen(req, timeout=10) as response:
                    data = json.loads(response.read().decode('utf-8'))
                    
                    for event in data[:20]:  # İlk 20 olay
                        impact = "HIGH" if event.get("impact") == "High" else "MEDIUM" if event.get("impact") == "Medium" else "LOW"
                        events.append({
                            "name": event.get("event"),
                            "code": self._extract_event_code(event.get("event", "")),
                            "date": event.get("date"),
                            "impact": impact,
                            "country": event.get("country"),
                            "previous": event.get("previous"),
                            "forecast": event.get("estimate"),
                            "actual": event.get("actual")
                        })
            except:
                events = sample_events
        else:
            events = sample_events
        
        self.cached_events = events
        self.last_fetch = datetime.now()
        
        return events
    
    def _extract_event_code(self, event_name: str) -> str:
        """Olay isminden kod çıkar."""
        name_lower = event_name.lower()
        
        if "fomc" in name_lower or "fed" in name_lower or "interest rate" in name_lower:
            return "FOMC"
        elif "nonfarm" in name_lower or "payroll" in name_lower:
            return "NFP"
        elif "cpi" in name_lower or "consumer price" in name_lower:
            return "CPI"
        elif "gdp" in name_lower:
            return "GDP"
        elif "unemploy" in name_lower:
            return "UNEMPLOYMENT"
        
        return "OTHER"
    
    def get_upcoming_high_impact(self, hours_ahead: int = 24) -> List[Dict]:
        """Önümüzdeki X saat içindeki yüksek etkili olayları getir."""
        if not self.cached_events or (datetime.now() - self.last_fetch).seconds > 3600:
            self.fetch_economic_calendar()
        
        now = datetime.now()
        cutoff = now + timedelta(hours=hours_ahead)
        
        high_impact = []
        for event in self.cached_events:
            try:
                event_date = datetime.fromisoformat(event['date'].replace('Z', ''))
                if now <= event_date <= cutoff and event['impact'] == 'HIGH':
                    time_until = (event_date - now).total_seconds() / 3600
                    event['hours_until'] = time_until
                    high_impact.append(event)
            except:
                pass
        
        return sorted(high_impact, key=lambda x: x.get('hours_until', 999))
    
    def should_avoid_trading(self, ticker: str = None) -> Tuple[bool, str]:
        """
        Volatility Guard: Trading'den kaçınılmalı mı?
        """
        upcoming = self.get_upcoming_high_impact(hours_ahead=4)
        
        for event in upcoming:
            code = event.get('code', '')
            if code in self.HIGH_IMPACT_EVENTS:
                event_info = self.HIGH_IMPACT_EVENTS[code]
                guard_hours = self.VOLATILITY_GUARD_HOURS.get(event_info['impact'], 1)
                
                if event.get('hours_until', 999) <= guard_hours:
                    return True, f"⚠️ VOLATILITY GUARD: {event['name']} {event['hours_until']:.1f} saat içinde! Trading önerilmez."
        
        return False, ""
    
    def analyze_impact(self, event: Dict) -> str:
        """Olayın potansiyel etkisini analiz et."""
        code = event.get('code', '')
        
        if code not in self.HIGH_IMPACT_EVENTS:
            return "Etki bilinmiyor"
        
        info = self.HIGH_IMPACT_EVENTS[code]
        assets = ", ".join(info['assets'])
        
        return f"Etkilenen varlıklar: {assets}"
    
    def generate_calendar_report(self, days: int = 7) -> str:
        """Ekonomik takvim raporu oluştur."""
        events = self.fetch_economic_calendar(days)
        
        report = f"""
<economic_calendar>
📅 EKONOMİK TAKVİM (Önümüzdeki {days} gün)
════════════════════════════════════════
"""
        high_impact = [e for e in events if e['impact'] == 'HIGH']
        
        if high_impact:
            report += "\n🔴 YÜKSEK ETKİLİ OLAYLAR:\n"
            for event in high_impact[:5]:
                try:
                    event_date = datetime.fromisoformat(event['date'].replace('Z', ''))
                    report += f"  • {event_date.strftime('%d/%m %H:%M')} - {event['name']}\n"
                    report += f"    Önceki: {event['previous']} | Beklenti: {event['forecast']}\n"
                except:
                    report += f"  • {event['name']}\n"
        else:
            report += "✅ Yüksek etkili olay yok\n"
        
        # Volatility Guard kontrolü
        should_avoid, reason = self.should_avoid_trading()
        if should_avoid:
            report += f"\n{reason}\n"
        
        report += "</economic_calendar>\n"
        return report
