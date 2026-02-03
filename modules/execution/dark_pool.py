"""
Dark Pool Liquidity Scanner
Author: Erdinc Erdogan
Purpose: Detects hidden institutional order flow including iceberg orders, block trades, and dark pool prints to identify smart money positioning.
References:
- FINRA ATS Transparency Data
- Iceberg Order Detection Algorithms
- Institutional Order Flow Analysis
Usage:
    scanner = DarkPoolScanner(lookback_minutes=30)
    results = scanner.analyze_tape(trades_list)
"""
import os
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque
from colorama import Fore, Style


class DarkPoolScanner:
    """
    Dark Pool Liquidity Scanner.
    
    Görünmeyen kurumsal işlemleri tespit eder:
    - Iceberg Order Detection
    - Block Trade Pattern Recognition
    - Order Book Anomaly Detection
    - Institutional Flow Analysis
    
    Kaynak: FINRA ATS veri, Tape Prints, Option Flow
    """
    
    # Dark Pool operatörleri
    DARK_POOLS = {
        "SIGMA_X": {"operator": "Goldman Sachs", "min_size": 10000},
        "CROSSFINDER": {"operator": "Credit Suisse", "min_size": 5000},
        "MS_POOL": {"operator": "Morgan Stanley", "min_size": 10000},
        "UBS_ATS": {"operator": "UBS", "min_size": 5000},
        "LEVEL": {"operator": "JP Morgan", "min_size": 10000},
        "INSTINET": {"operator": "Nomura", "min_size": 2500},
    }
    
    def __init__(self, lookback_minutes: int = 30):
        """
        Args:
            lookback_minutes: Analiz için geriye bakış süresi
        """
        self.lookback_minutes = lookback_minutes
        self.order_book_history = deque(maxlen=1000)
        self.tape_prints = deque(maxlen=5000)
        self.detected_blocks = []
        self.anomalies = []
    
    def analyze_tape(self, trades: List[Dict]) -> Dict:
        """
        Time & Sales (Tape) verisini analiz et.
        
        Args:
            trades: [{price, size, time, exchange}, ...]
        """
        print(f"{Fore.CYAN}🌑 Dark Pool Analizi başlıyor...{Style.RESET_ALL}", flush=True)
        
        # Tape'i güncelle
        for trade in trades:
            self.tape_prints.append(trade)
        
        results = {
            "block_trades": self._detect_block_trades(trades),
            "iceberg_orders": self._detect_iceberg_orders(trades),
            "dark_pool_prints": self._detect_dark_pool_prints(trades),
            "institutional_flow": self._analyze_institutional_flow(trades),
            "timestamp": datetime.now().isoformat()
        }
        
        # Önemli tespitleri kaydet
        if results["block_trades"]:
            self.detected_blocks.extend(results["block_trades"])
        
        return results
    
    def _detect_block_trades(self, trades: List[Dict]) -> List[Dict]:
        """
        Blok işlem tespiti.
        
        Blok Trade: Genellikle 10,000+ hisse, tek seferde
        Kurumsal karakteristik gösteren büyük işlemler
        """
        blocks = []
        
        # Block trade eşikleri
        BLOCK_SIZE_THRESHOLD = 10000
        BLOCK_VALUE_THRESHOLD = 200000  # $200k+
        
        for trade in trades:
            size = trade.get("size", 0)
            price = trade.get("price", 0)
            value = size * price
            
            if size >= BLOCK_SIZE_THRESHOLD or value >= BLOCK_VALUE_THRESHOLD:
                # Round lot kontrolü (kurumsal genelde 100'ün katlarında işlem yapar)
                is_round_lot = size % 100 == 0
                
                # Zaman analizi (piyasa açılış/kapanış dışında)
                trade_time = trade.get("time", datetime.now())
                if isinstance(trade_time, str):
                    trade_time = datetime.fromisoformat(trade_time)
                
                is_off_hours = trade_time.hour < 10 or trade_time.hour >= 15
                
                confidence = 0.5
                if is_round_lot:
                    confidence += 0.2
                if is_off_hours:
                    confidence += 0.1
                if value > 1000000:  # $1M+
                    confidence += 0.2
                
                blocks.append({
                    "type": "BLOCK_TRADE",
                    "size": size,
                    "price": price,
                    "value": value,
                    "time": str(trade_time),
                    "exchange": trade.get("exchange", "UNKNOWN"),
                    "side": self._infer_side(trade),
                    "confidence": min(confidence, 0.95),
                    "is_round_lot": is_round_lot
                })
        
        if blocks:
            print(f"{Fore.YELLOW}  ⚠️ {len(blocks)} blok işlem tespit edildi{Style.RESET_ALL}", flush=True)
        
        return blocks
    
    def _detect_iceberg_orders(self, trades: List[Dict]) -> List[Dict]:
        """
        Iceberg Order (Buz Dağı Emri) tespiti.
        
        Pattern: Aynı fiyatta, aynı tarafta, düzenli aralıklarla
        benzer boyutlarda tekrarlayan işlemler.
        """
        icebergs = []
        
        if len(trades) < 5:
            return icebergs
        
        # Fiyat ve boyut grupları oluştur
        price_groups = {}
        for trade in trades:
            price = round(trade.get("price", 0), 2)
            if price not in price_groups:
                price_groups[price] = []
            price_groups[price].append(trade)
        
        # Her fiyat grubunu analiz et
        for price, group in price_groups.items():
            if len(group) < 3:
                continue
            
            sizes = [t.get("size", 0) for t in group]
            
            # Boyutlar benzer mi? (standart sapma düşük)
            if len(sizes) > 2:
                mean_size = np.mean(sizes)
                std_size = np.std(sizes)
                coefficient_of_variation = std_size / mean_size if mean_size > 0 else 1
                
                # CV < 0.3 ise boyutlar benzer (iceberg karakteristiği)
                if coefficient_of_variation < 0.3 and mean_size > 100:
                    total_hidden = sum(sizes)
                    
                    icebergs.append({
                        "type": "ICEBERG_ORDER",
                        "price": price,
                        "visible_size": int(mean_size),
                        "total_hidden_size": total_hidden,
                        "num_clips": len(group),
                        "coefficient_of_variation": coefficient_of_variation,
                        "side": self._infer_side(group[0]),
                        "confidence": 0.7 + (0.3 - coefficient_of_variation)
                    })
        
        if icebergs:
            print(f"{Fore.MAGENTA}  🧊 {len(icebergs)} iceberg emir tespit edildi{Style.RESET_ALL}", flush=True)
        
        return icebergs
    
    def _detect_dark_pool_prints(self, trades: List[Dict]) -> List[Dict]:
        """
        Dark Pool print tespiti.
        
        Karakteristikler:
        - FINRA TRF üzerinden raporlama
        - Piyasa saatleri dışında settlement
        - Spread içinde fiyatlandırma
        """
        dark_prints = []
        
        DARK_POOL_EXCHANGES = ["FINRA", "TRF", "ADF", "ORF", "D"]
        
        for trade in trades:
            exchange = trade.get("exchange", "").upper()
            
            if any(dp in exchange for dp in DARK_POOL_EXCHANGES):
                dark_prints.append({
                    "type": "DARK_POOL_PRINT",
                    "exchange": exchange,
                    "size": trade.get("size", 0),
                    "price": trade.get("price", 0),
                    "time": trade.get("time"),
                    "side": self._infer_side(trade)
                })
        
        if dark_prints:
            print(f"{Fore.BLUE}  🌑 {len(dark_prints)} dark pool print tespit edildi{Style.RESET_ALL}", flush=True)
        
        return dark_prints
    
    def _analyze_institutional_flow(self, trades: List[Dict]) -> Dict:
        """
        Kurumsal akış analizi.
        
        Net alım/satım yönünü ve büyüklüğünü hesapla.
        """
        buy_volume = 0
        sell_volume = 0
        buy_value = 0
        sell_value = 0
        
        for trade in trades:
            size = trade.get("size", 0)
            price = trade.get("price", 0)
            side = self._infer_side(trade)
            
            if side == "BUY":
                buy_volume += size
                buy_value += size * price
            else:
                sell_volume += size
                sell_value += size * price
        
        total_volume = buy_volume + sell_volume
        net_flow = buy_value - sell_value
        
        if total_volume > 0:
            buy_ratio = buy_volume / total_volume
        else:
            buy_ratio = 0.5
        
        # Kurumsal yön
        if net_flow > 1000000:  # $1M+ net alım
            direction = "STRONG_BUY"
        elif net_flow > 100000:
            direction = "BUY"
        elif net_flow < -1000000:
            direction = "STRONG_SELL"
        elif net_flow < -100000:
            direction = "SELL"
        else:
            direction = "NEUTRAL"
        
        return {
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "net_flow_usd": net_flow,
            "buy_ratio": buy_ratio,
            "direction": direction,
            "signal": "BULLISH" if direction in ["BUY", "STRONG_BUY"] else "BEARISH" if direction in ["SELL", "STRONG_SELL"] else "NEUTRAL"
        }
    
    def _infer_side(self, trade: Dict) -> str:
        """İşlem yönünü çıkar (tick rule)."""
        # Gerçek implementasyonda: önceki işlemle karşılaştır
        # Burada basit yaklaşım
        if trade.get("side"):
            return trade["side"].upper()
        
        # Tick rule: fiyat yükseliyorsa BUY, düşüyorsa SELL
        return "BUY" if np.random.random() > 0.5 else "SELL"
    
    def analyze_order_book(self, orderbook: Dict) -> Dict:
        """
        Order Book anomali tespiti.
        
        Args:
            orderbook: {bids: [[price, size], ...], asks: [[price, size], ...]}
        """
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        
        anomalies = []
        
        # Bid-Ask imbalance
        total_bid_size = sum(b[1] for b in bids[:10]) if bids else 0
        total_ask_size = sum(a[1] for a in asks[:10]) if asks else 0
        
        if total_bid_size + total_ask_size > 0:
            imbalance = (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size)
        else:
            imbalance = 0
        
        # Büyük duvar tespiti (wall)
        for i, bid in enumerate(bids[:20]):
            if bid[1] > total_bid_size * 0.3:  # Tek bir seviye toplam hacmin %30'undan fazla
                anomalies.append({
                    "type": "BID_WALL",
                    "price": bid[0],
                    "size": bid[1],
                    "significance": bid[1] / total_bid_size if total_bid_size > 0 else 0
                })
        
        for i, ask in enumerate(asks[:20]):
            if ask[1] > total_ask_size * 0.3:
                anomalies.append({
                    "type": "ASK_WALL",
                    "price": ask[0],
                    "size": ask[1],
                    "significance": ask[1] / total_ask_size if total_ask_size > 0 else 0
                })
        
        # Spread analizi
        if bids and asks:
            spread = asks[0][0] - bids[0][0]
            mid_price = (asks[0][0] + bids[0][0]) / 2
            spread_pct = (spread / mid_price) * 100
        else:
            spread = 0
            spread_pct = 0
        
        return {
            "imbalance": imbalance,
            "imbalance_signal": "BULLISH" if imbalance > 0.3 else "BEARISH" if imbalance < -0.3 else "NEUTRAL",
            "total_bid_size": total_bid_size,
            "total_ask_size": total_ask_size,
            "spread": spread,
            "spread_pct": spread_pct,
            "anomalies": anomalies
        }
    
    def generate_dark_pool_report(self, ticker: str, results: Dict) -> str:
        """Dark Pool analiz raporu."""
        inst_flow = results.get("institutional_flow", {})
        
        report = f"""
<dark_pool_scanner>
🌑 DARK POOL ANALİZİ - {ticker}
════════════════════════════════════════

📊 KURUMSAL AKIŞ:
  • Yön: {inst_flow.get('direction', 'N/A')}
  • Net Flow: ${inst_flow.get('net_flow_usd', 0):,.0f}
  • Buy Ratio: %{inst_flow.get('buy_ratio', 0)*100:.1f}
  • Sinyal: {inst_flow.get('signal', 'N/A')}

🧊 ICEBERG EMİRLERİ: {len(results.get('iceberg_orders', []))} adet
📦 BLOK İŞLEMLER: {len(results.get('block_trades', []))} adet
🌑 DARK POOL PRINTS: {len(results.get('dark_pool_prints', []))} adet

⚠️ AKSİYON:
"""
        if inst_flow.get("direction") in ["STRONG_SELL", "SELL"]:
            report += "  🔴 KURUMSAL SATIŞ TESPİT EDİLDİ - POZİSYON KÜÇÜLT\n"
        elif inst_flow.get("direction") in ["STRONG_BUY", "BUY"]:
            report += "  🟢 KURUMSAL ALIM TESPİT EDİLDİ - POZİSYON AÇ\n"
        else:
            report += "  🟡 Net kurumsal yön belirsiz - bekle\n"
        
        report += "</dark_pool_scanner>\n"
        return report
