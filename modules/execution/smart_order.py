"""
Smart Order Router (SOR)
Author: Erdinc Erdogan
Purpose: Routes large orders using TWAP, VWAP, and Iceberg strategies to minimize market impact and achieve optimal execution prices.
References:
- TWAP/VWAP Execution Algorithms
- Iceberg Order Strategies
- Smart Order Routing Best Practices
Usage:
    router = SmartOrderRouter()
    order = router.create_twap_order("AAPL", "BUY", total_quantity=10000, duration_minutes=60)
"""
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional
from colorama import Fore, Style


class SmartOrderRouter:
    """
    Smart Order Routing (SOR) - Akıllı Emir Yönlendirme.
    
    Büyük emirleri parçalara bölerek piyasa etkisini (slippage) azaltır.
    
    Stratejiler:
    - TWAP: Time-Weighted Average Price
    - VWAP: Volume-Weighted Average Price
    - Iceberg: Büyük emirleri küçük parçalar halinde göster
    """
    
    def __init__(self, 
                 execute_callback: Callable = None,
                 get_price_callback: Callable = None,
                 get_volume_callback: Callable = None):
        """
        Args:
            execute_callback: Gerçek emir gönderme fonksiyonu
            get_price_callback: Anlık fiyat alma fonksiyonu
            get_volume_callback: Anlık hacim alma fonksiyonu
        """
        self.execute_callback = execute_callback or self._mock_execute
        self.get_price_callback = get_price_callback or self._mock_price
        self.get_volume_callback = get_volume_callback or self._mock_volume
        
        self.active_orders = {}
        self.completed_orders = []
        self._lock = threading.Lock()
    
    def _mock_execute(self, ticker: str, side: str, quantity: float, price: float) -> Dict:
        """Mock emir gönderimi."""
        return {
            "order_id": f"ORD-{int(time.time()*1000)}",
            "ticker": ticker,
            "side": side,
            "quantity": quantity,
            "price": price,
            "status": "FILLED",
            "filled_at": datetime.now().isoformat()
        }
    
    def _mock_price(self, ticker: str) -> float:
        """Mock fiyat."""
        import random
        base_prices = {"AAPL": 180, "MSFT": 400, "NVDA": 500, "BTC": 40000}
        base = base_prices.get(ticker, 100)
        return base * (1 + random.uniform(-0.001, 0.001))  # ±0.1% volatilite
    
    def _mock_volume(self, ticker: str) -> float:
        """Mock hacim."""
        import random
        return random.uniform(100000, 500000)
    
    def create_twap_order(self,
                         ticker: str,
                         side: str,
                         total_quantity: float,
                         duration_minutes: int = 60,
                         num_slices: int = 10) -> Dict:
        """
        TWAP (Time-Weighted Average Price) emri oluştur.
        
        Emri eşit zaman dilimlerine bölerek gönderir.
        
        Args:
            ticker: Sembol
            side: "BUY" veya "SELL"
            total_quantity: Toplam miktar
            duration_minutes: Toplam süre (dakika)
            num_slices: Kaç parçaya bölüneceği
        """
        print(f"{Fore.CYAN}📊 TWAP Emri Oluşturuluyor...{Style.RESET_ALL}", flush=True)
        print(f"   {side} {total_quantity} {ticker} | {duration_minutes} dk | {num_slices} dilim", flush=True)
        
        slice_quantity = total_quantity / num_slices
        interval_seconds = (duration_minutes * 60) / num_slices
        
        order = {
            "order_id": f"TWAP-{int(time.time()*1000)}",
            "type": "TWAP",
            "ticker": ticker,
            "side": side,
            "total_quantity": total_quantity,
            "remaining_quantity": total_quantity,
            "slice_quantity": slice_quantity,
            "num_slices": num_slices,
            "slices_executed": 0,
            "interval_seconds": interval_seconds,
            "duration_minutes": duration_minutes,
            "created_at": datetime.now(),
            "status": "ACTIVE",
            "fills": [],
            "average_price": 0.0,
            "total_cost": 0.0
        }
        
        with self._lock:
            self.active_orders[order["order_id"]] = order
        
        # Background thread ile çalıştır
        thread = threading.Thread(target=self._execute_twap, args=(order["order_id"],))
        thread.daemon = True
        thread.start()
        
        return order
    
    def _execute_twap(self, order_id: str):
        """TWAP emrini dilim dilim çalıştır."""
        order = self.active_orders.get(order_id)
        if not order:
            return
        
        for i in range(order["num_slices"]):
            if order["status"] != "ACTIVE":
                break
            
            current_price = self.get_price_callback(order["ticker"])
            
            result = self.execute_callback(
                order["ticker"],
                order["side"],
                order["slice_quantity"],
                current_price
            )
            
            with self._lock:
                order["fills"].append({
                    "slice": i + 1,
                    "quantity": order["slice_quantity"],
                    "price": current_price,
                    "timestamp": datetime.now().isoformat()
                })
                order["slices_executed"] += 1
                order["remaining_quantity"] -= order["slice_quantity"]
                order["total_cost"] += order["slice_quantity"] * current_price
                order["average_price"] = order["total_cost"] / (order["slice_quantity"] * order["slices_executed"])
                
                print(f"{Fore.GREEN}   ✓ TWAP Dilim {i+1}/{order['num_slices']}: {order['slice_quantity']} @ ${current_price:.2f}{Style.RESET_ALL}", flush=True)
            
            if i < order["num_slices"] - 1:
                time.sleep(order["interval_seconds"])
        
        with self._lock:
            order["status"] = "COMPLETED"
            order["completed_at"] = datetime.now()
            self.completed_orders.append(order)
            print(f"{Fore.GREEN}✅ TWAP Tamamlandı: Ort. Fiyat ${order['average_price']:.2f}{Style.RESET_ALL}", flush=True)
    
    def create_vwap_order(self,
                         ticker: str,
                         side: str,
                         total_quantity: float,
                         duration_minutes: int = 60,
                         participation_rate: float = 0.1) -> Dict:
        """
        VWAP (Volume-Weighted Average Price) emri oluştur.
        
        Piyasa hacmine göre emir boyutunu dinamik ayarlar.
        
        Args:
            ticker: Sembol
            side: "BUY" veya "SELL"
            total_quantity: Toplam miktar
            duration_minutes: Toplam süre
            participation_rate: Piyasa hacminin yüzde kaçına katılınacak (0.1 = %10)
        """
        print(f"{Fore.CYAN}📊 VWAP Emri Oluşturuluyor...{Style.RESET_ALL}", flush=True)
        print(f"   {side} {total_quantity} {ticker} | Participation: %{participation_rate*100}", flush=True)
        
        order = {
            "order_id": f"VWAP-{int(time.time()*1000)}",
            "type": "VWAP",
            "ticker": ticker,
            "side": side,
            "total_quantity": total_quantity,
            "remaining_quantity": total_quantity,
            "participation_rate": participation_rate,
            "duration_minutes": duration_minutes,
            "created_at": datetime.now(),
            "status": "ACTIVE",
            "fills": [],
            "average_price": 0.0,
            "total_cost": 0.0,
            "volume_weighted_price": 0.0
        }
        
        with self._lock:
            self.active_orders[order["order_id"]] = order
        
        thread = threading.Thread(target=self._execute_vwap, args=(order["order_id"],))
        thread.daemon = True
        thread.start()
        
        return order
    
    def _execute_vwap(self, order_id: str):
        """VWAP emrini hacme göre çalıştır."""
        order = self.active_orders.get(order_id)
        if not order:
            return
        
        check_interval = 30  # Her 30 saniyede kontrol
        max_checks = (order["duration_minutes"] * 60) // check_interval
        total_volume_weighted = 0.0
        total_volume = 0.0
        
        for i in range(max_checks):
            if order["status"] != "ACTIVE" or order["remaining_quantity"] <= 0:
                break
            
            current_price = self.get_price_callback(order["ticker"])
            current_volume = self.get_volume_callback(order["ticker"])
            
            # Participation rate'e göre emir boyutu
            slice_quantity = min(
                current_volume * order["participation_rate"],
                order["remaining_quantity"]
            )
            
            if slice_quantity > 0:
                result = self.execute_callback(
                    order["ticker"],
                    order["side"],
                    slice_quantity,
                    current_price
                )
                
                with self._lock:
                    order["fills"].append({
                        "iteration": i + 1,
                        "quantity": slice_quantity,
                        "price": current_price,
                        "market_volume": current_volume,
                        "timestamp": datetime.now().isoformat()
                    })
                    order["remaining_quantity"] -= slice_quantity
                    order["total_cost"] += slice_quantity * current_price
                    
                    # VWAP hesaplama
                    total_volume_weighted += current_price * current_volume
                    total_volume += current_volume
                    order["volume_weighted_price"] = total_volume_weighted / total_volume if total_volume > 0 else 0
                    
                    filled_qty = order["total_quantity"] - order["remaining_quantity"]
                    order["average_price"] = order["total_cost"] / filled_qty if filled_qty > 0 else 0
                    
                    print(f"{Fore.GREEN}   ✓ VWAP: {slice_quantity:.2f} @ ${current_price:.2f} (Market VWAP: ${order['volume_weighted_price']:.2f}){Style.RESET_ALL}", flush=True)
            
            if i < max_checks - 1:
                time.sleep(check_interval)
        
        with self._lock:
            order["status"] = "COMPLETED"
            order["completed_at"] = datetime.now()
            self.completed_orders.append(order)
            
            # Performans değerlendirmesi
            slippage = ((order["average_price"] - order["volume_weighted_price"]) / order["volume_weighted_price"]) * 100 if order["volume_weighted_price"] > 0 else 0
            print(f"{Fore.GREEN}✅ VWAP Tamamlandı: Ort ${order['average_price']:.2f} vs Market VWAP ${order['volume_weighted_price']:.2f} (Slippage: {slippage:+.3f}%){Style.RESET_ALL}", flush=True)
    
    def create_iceberg_order(self,
                            ticker: str,
                            side: str,
                            total_quantity: float,
                            visible_quantity: float) -> Dict:
        """
        Iceberg emri oluştur.
        
        Büyük emrin sadece küçük bir kısmını piyasada gösterir.
        
        Args:
            ticker: Sembol
            side: "BUY" veya "SELL"
            total_quantity: Toplam miktar
            visible_quantity: Görünür miktar (her seferinde)
        """
        print(f"{Fore.CYAN}🧊 Iceberg Emri: Toplam {total_quantity}, Görünür {visible_quantity}{Style.RESET_ALL}", flush=True)
        
        order = {
            "order_id": f"ICE-{int(time.time()*1000)}",
            "type": "ICEBERG",
            "ticker": ticker,
            "side": side,
            "total_quantity": total_quantity,
            "visible_quantity": visible_quantity,
            "hidden_quantity": total_quantity - visible_quantity,
            "remaining_quantity": total_quantity,
            "status": "ACTIVE",
            "fills": []
        }
        
        return order
    
    def cancel_order(self, order_id: str) -> bool:
        """Aktif emri iptal et."""
        with self._lock:
            if order_id in self.active_orders:
                self.active_orders[order_id]["status"] = "CANCELLED"
                print(f"{Fore.YELLOW}🚫 Emir iptal edildi: {order_id}{Style.RESET_ALL}", flush=True)
                return True
        return False
    
    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Emir durumunu getir."""
        return self.active_orders.get(order_id) or next(
            (o for o in self.completed_orders if o["order_id"] == order_id), None
        )
    
    def generate_sor_report(self) -> str:
        """SOR performans raporu oluştur."""
        report = """
<smart_order_routing>
🔀 SMART ORDER ROUTING RAPORU
════════════════════════════════════════

"""
        for order in self.completed_orders[-5:]:
            fills_count = len(order.get("fills", []))
            report += f"""📋 {order['type']} - {order['order_id'][-8:]}
   • {order['side']} {order['total_quantity']} {order['ticker']}
   • Ort. Fiyat: ${order.get('average_price', 0):.2f}
   • Dilim Sayısı: {fills_count}
   • Durum: {order['status']}

"""
        
        report += "</smart_order_routing>\n"
        return report
