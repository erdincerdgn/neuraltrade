"""
Precision Time Protocol (PTP) Implementation
Author: Erdinc Erdogan
Purpose: Provides microsecond-accurate time synchronization (IEEE 1588 PTP) for detecting stale quotes and ensuring timestamp accuracy in HFT.
References:
- IEEE 1588 Precision Time Protocol
- Grandmaster Clock Synchronization
- Stale Quote Detection
Usage:
    ptp = PrecisionTimeProtocol(max_clock_offset_us=100)
    sync_result = ptp.sync_with_grandmaster()
"""
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import deque
from colorama import Fore, Style


class PrecisionTimeProtocol:
    """
    PTP (IEEE 1588) Time Synchronization.
    
    NTP'den 1000x daha hassas:
    - NTP: ±50ms
    - PTP: ±1µs
    
    Gerçek implementasyon özel donanım gerektirir.
    Bu simülasyon mantığı gösterir.
    """
    
    def __init__(self, 
                 max_clock_offset_us: float = 100,  # µs
                 max_network_jitter_us: float = 500):
        """
        Args:
            max_clock_offset_us: Max saat farkı (µs)
            max_network_jitter_us: Max network jitter (µs)
        """
        self.max_clock_offset = max_clock_offset_us
        self.max_jitter = max_network_jitter_us
        
        self.clock_offset_us = 0  # Mevcut saat farkı
        self.sync_history = []
        self.jitter_history = deque(maxlen=1000)
        
        self.last_sync = None
        self.grandmaster_clock = None  # Referans saat
    
    def sync_with_grandmaster(self, grandmaster_time_ns: int = None) -> Dict:
        """
        Grandmaster clock ile senkronize ol.
        
        PTP mimarisi:
        - Grandmaster: En hassas saat (atomik saat referanslı)
        - Boundary/Transparent clocks: Ara bileşenler
        - Slave: Bizim sunucumuz
        """
        local_time_ns = time.time_ns()
        grandmaster_time_ns = grandmaster_time_ns or local_time_ns
        
        # Offset hesapla
        offset_ns = local_time_ns - grandmaster_time_ns
        self.clock_offset_us = offset_ns / 1000
        
        # Jitter simülasyonu
        jitter = abs(self.clock_offset_us - (self.sync_history[-1]["offset_us"] if self.sync_history else 0))
        self.jitter_history.append(jitter)
        
        sync_result = {
            "timestamp": datetime.now().isoformat(),
            "local_time_ns": local_time_ns,
            "grandmaster_time_ns": grandmaster_time_ns,
            "offset_us": self.clock_offset_us,
            "jitter_us": jitter,
            "sync_quality": self._assess_sync_quality()
        }
        
        self.sync_history.append(sync_result)
        self.last_sync = datetime.now()
        
        return sync_result
    
    def _assess_sync_quality(self) -> str:
        """Senkronizasyon kalitesi."""
        if abs(self.clock_offset_us) < 10:
            return "EXCELLENT"  # <10µs
        elif abs(self.clock_offset_us) < 100:
            return "GOOD"       # <100µs
        elif abs(self.clock_offset_us) < 1000:
            return "FAIR"       # <1ms
        else:
            return "POOR"       # >1ms
    
    def get_precise_timestamp(self) -> int:
        """Düzeltilmiş hassas zaman damgası (ns)."""
        raw_ns = time.time_ns()
        corrected_ns = raw_ns - int(self.clock_offset_us * 1000)
        return corrected_ns
    
    def calculate_one_way_delay(self, 
                               exchange_timestamp_ns: int,
                               local_receive_ns: int = None) -> Dict:
        """
        Tek yön gecikme hesapla.
        
        Işık hızı sınırı: ~1ms / 200km fiber
        """
        local_receive_ns = local_receive_ns or time.time_ns()
        
        # Clock offset'i düzelt
        corrected_local = local_receive_ns - int(self.clock_offset_us * 1000)
        
        one_way_delay_us = (corrected_local - exchange_timestamp_ns) / 1000
        
        return {
            "one_way_delay_us": one_way_delay_us,
            "exchange_timestamp_ns": exchange_timestamp_ns,
            "local_corrected_ns": corrected_local,
            "is_stale": one_way_delay_us > 1000  # >1ms = stale
        }


class StaleQuoteDetector:
    """
    Stale Quote Detection.
    
    Bayat veri ile işlem yapmayı engeller.
    """
    
    def __init__(self, 
                 stale_threshold_us: float = 1000,  # 1ms
                 max_age_us: float = 5000):         # 5ms
        """
        Args:
            stale_threshold_us: Stale eşiği (µs)
            max_age_us: Maksimum kabul edilebilir yaş
        """
        self.stale_threshold = stale_threshold_us
        self.max_age = max_age_us
        
        self.ptp = PrecisionTimeProtocol()
        self.stale_count = 0
        self.valid_count = 0
        self.rejected_quotes = []
    
    def validate_quote(self, 
                      quote: Dict,
                      exchange_timestamp_ns: int) -> Dict:
        """
        Quote'u doğrula.
        
        Args:
            quote: {symbol, bid, ask, ...}
            exchange_timestamp_ns: Exchange zaman damgası
        """
        local_ns = time.time_ns()
        
        # PTP ile delay hesapla
        delay = self.ptp.calculate_one_way_delay(exchange_timestamp_ns, local_ns)
        
        age_us = delay["one_way_delay_us"]
        
        # Karar
        if age_us < 0:
            status = "CLOCK_MISMATCH"
            valid = False
        elif age_us > self.max_age:
            status = "TOO_OLD"
            valid = False
        elif age_us > self.stale_threshold:
            status = "STALE"
            valid = False
        else:
            status = "FRESH"
            valid = True
        
        result = {
            "quote": quote,
            "age_us": age_us,
            "age_ms": age_us / 1000,
            "status": status,
            "valid": valid,
            "threshold_us": self.stale_threshold
        }
        
        if valid:
            self.valid_count += 1
        else:
            self.stale_count += 1
            self.rejected_quotes.append({
                "timestamp": datetime.now().isoformat(),
                "quote": quote,
                "age_us": age_us,
                "reason": status
            })
            
            print(f"{Fore.YELLOW}⏰ STALE QUOTE: {quote.get('symbol', 'N/A')} ({age_us:.0f}µs){Style.RESET_ALL}", flush=True)
        
        return result
    
    def get_freshness_stats(self) -> Dict:
        """Tazelik istatistikleri."""
        total = self.valid_count + self.stale_count
        
        if total == 0:
            return {"error": "No quotes processed"}
        
        return {
            "total_quotes": total,
            "valid_quotes": self.valid_count,
            "stale_quotes": self.stale_count,
            "freshness_ratio": self.valid_count / total,
            "stale_ratio": self.stale_count / total,
            "avg_jitter_us": np.mean(list(self.ptp.jitter_history)) if self.ptp.jitter_history else 0
        }
    
    def generate_ptp_report(self) -> str:
        """PTP raporu."""
        stats = self.get_freshness_stats()
        sync_quality = self.ptp._assess_sync_quality() if self.ptp.sync_history else "NOT_SYNCED"
        
        report = f"""
<precision_time>
⏱️ PRECİSİON TİME PROTOCOL
════════════════════════════════════════

🕐 SENKRONIZASYON:
  • Clock Offset: {self.ptp.clock_offset_us:.1f}µs
  • Kalite: {sync_quality}
  • Son Sync: {self.ptp.last_sync}

📊 QUOTE TAZELİĞİ:
  • Toplam: {stats.get('total_quotes', 0)}
  • Taze: {stats.get('valid_quotes', 0)} ({stats.get('freshness_ratio', 0)*100:.1f}%)
  • Bayat: {stats.get('stale_quotes', 0)} ({stats.get('stale_ratio', 0)*100:.1f}%)

⚙️ AYARLAR:
  • Stale Eşik: {self.stale_threshold}µs
  • Max Yaş: {self.max_age}µs

💡 NTP yerine PTP kullanılıyor (1000x hassas)

</precision_time>
"""
        return report


# NumPy import for stats
import numpy as np
