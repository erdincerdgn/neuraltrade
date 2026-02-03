"""
ERC-4626 Tokenized Strategy Vault
Author: Erdinc Erdogan
Purpose: Implements trustless hedge fund management with tokenized shares, automatic fee distribution, and high-water mark performance tracking.
References:
- ERC-4626 Tokenized Vault Standard
- DeFi 2.0 Asset Management
- Performance Fee with High-Water Mark
Usage:
    vault = TokenizedVault(vault_name='Alpha Vault', base_token='USDC')
    deposit = vault.deposit(investor='0x...', amount=10000)
    withdrawal = vault.withdraw(investor='0x...', shares=100)
"""
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Optional
from colorama import Fore, Style


class TokenizedVault:
    """
    ERC-4626 Strategy Vault.
    
    DeFi 2.0: Trustless hedge fund yönetimi.
    - Yatırımcılar USDC yatırır
    - Bot stratejiyi yönetir
    - Akıllı kontrat kârı otomatik dağıtır
    """
    
    # Fee yapısı
    MANAGEMENT_FEE_PCT = 2.0    # Yıllık yönetim ücreti
    PERFORMANCE_FEE_PCT = 20.0  # Kâr payı
    
    def __init__(self, 
                 vault_name: str = "NeuralTrade Alpha Vault",
                 base_token: str = "USDC"):
        self.vault_name = vault_name
        self.base_token = base_token
        self.vault_token = f"nt{vault_name[:3].upper()}"  # ntALPHA
        
        # Vault state
        self.total_assets = 0
        self.total_shares = 0
        self.share_price = 1.0
        self.high_water_mark = 1.0
        
        # Investor tracking
        self.investors = {}  # address -> shares
        self.deposits = []
        self.withdrawals = []
        
        # Performance
        self.nav_history = []
        self.fees_collected = 0
        self.created_at = datetime.now()
    
    def deposit(self, investor: str, amount: float) -> Dict:
        """
        Vault'a yatırım.
        
        Args:
            investor: Yatırımcı adresi
            amount: USDC miktarı
        """
        if amount <= 0:
            return {"error": "Invalid amount"}
        
        # Share hesapla
        shares = amount / self.share_price if self.share_price > 0 else amount
        
        # State güncelle
        self.total_assets += amount
        self.total_shares += shares
        
        if investor not in self.investors:
            self.investors[investor] = 0
        self.investors[investor] += shares
        
        deposit = {
            "investor": investor,
            "amount": amount,
            "shares": shares,
            "share_price": self.share_price,
            "timestamp": datetime.now().isoformat(),
            "tx_hash": self._generate_tx_hash(investor, amount)
        }
        self.deposits.append(deposit)
        
        print(f"{Fore.GREEN}💰 Deposit: {investor[:10]}... → {amount} {self.base_token} = {shares:.4f} shares{Style.RESET_ALL}", flush=True)
        
        return deposit
    
    def withdraw(self, investor: str, shares: float = None) -> Dict:
        """
        Vault'dan çekim.
        
        Args:
            investor: Yatırımcı adresi
            shares: Pay miktarı (None = tamamı)
        """
        if investor not in self.investors:
            return {"error": "Investor not found"}
        
        available_shares = self.investors[investor]
        shares = shares or available_shares
        
        if shares > available_shares:
            return {"error": "Insufficient shares"}
        
        # USDC hesapla
        amount = shares * self.share_price
        
        # State güncelle
        self.total_assets -= amount
        self.total_shares -= shares
        self.investors[investor] -= shares
        
        if self.investors[investor] == 0:
            del self.investors[investor]
        
        withdrawal = {
            "investor": investor,
            "shares": shares,
            "amount": amount,
            "share_price": self.share_price,
            "timestamp": datetime.now().isoformat(),
            "tx_hash": self._generate_tx_hash(investor, amount)
        }
        self.withdrawals.append(withdrawal)
        
        print(f"{Fore.YELLOW}💸 Withdraw: {investor[:10]}... ← {amount:.2f} {self.base_token}{Style.RESET_ALL}", flush=True)
        
        return withdrawal
    
    def update_nav(self, new_total_assets: float):
        """
        NAV güncelle (trading sonuçları sonrası).
        
        Args:
            new_total_assets: Yeni toplam varlık değeri
        """
        old_nav = self.share_price
        
        # Performance fee hesapla
        profit = new_total_assets - self.total_assets
        
        if profit > 0 and self.share_price > self.high_water_mark:
            # High water mark aşıldı = performance fee
            excess_profit = profit * self.PERFORMANCE_FEE_PCT / 100
            new_total_assets -= excess_profit
            self.fees_collected += excess_profit
            self.high_water_mark = new_total_assets / self.total_shares if self.total_shares > 0 else 1
        
        # Update
        self.total_assets = new_total_assets
        self.share_price = new_total_assets / self.total_shares if self.total_shares > 0 else 1
        
        self.nav_history.append({
            "timestamp": datetime.now().isoformat(),
            "share_price": self.share_price,
            "total_assets": self.total_assets,
            "change_pct": ((self.share_price - old_nav) / old_nav) * 100 if old_nav > 0 else 0
        })
    
    def get_investor_balance(self, investor: str) -> Dict:
        """Yatırımcı bakiyesi."""
        shares = self.investors.get(investor, 0)
        value = shares * self.share_price
        
        return {
            "investor": investor,
            "shares": shares,
            "value_usdc": value,
            "share_price": self.share_price,
            "pct_of_vault": (shares / self.total_shares * 100) if self.total_shares > 0 else 0
        }
    
    def _generate_tx_hash(self, investor: str, amount: float) -> str:
        """Tx hash simülasyonu."""
        data = f"{investor}{amount}{time.time()}"
        return "0x" + hashlib.sha256(data.encode()).hexdigest()[:64]
    
    def get_vault_stats(self) -> Dict:
        """Vault istatistikleri."""
        # ROI hesapla
        if self.nav_history:
            initial_price = 1.0
            current_price = self.share_price
            roi = ((current_price - initial_price) / initial_price) * 100
        else:
            roi = 0
        
        return {
            "vault_name": self.vault_name,
            "vault_token": self.vault_token,
            "total_assets_usdc": self.total_assets,
            "total_shares": self.total_shares,
            "share_price": self.share_price,
            "high_water_mark": self.high_water_mark,
            "num_investors": len(self.investors),
            "roi_pct": roi,
            "fees_collected": self.fees_collected,
            "age_days": (datetime.now() - self.created_at).days
        }
    
    def generate_vault_report(self) -> str:
        """Vault raporu."""
        stats = self.get_vault_stats()
        
        report = f"""
<tokenized_vault>
🏦 TOKENIZED VAULT: {stats['vault_name']}
════════════════════════════════════════

💰 AUM (Assets Under Management):
  • Toplam Varlık: ${stats['total_assets_usdc']:,.2f} USDC
  • Share Price: ${stats['share_price']:.4f}
  • High Water Mark: ${stats['high_water_mark']:.4f}

👥 YATIRIMCILAR:
  • Sayı: {stats['num_investors']}
  • Toplam Pay: {stats['total_shares']:,.4f}

📈 PERFORMANS:
  • ROI: %{stats['roi_pct']:.2f}
  • Toplanan Fee: ${stats['fees_collected']:.2f}

⚙️ FEE YAPISI:
  • Yönetim: %{self.MANAGEMENT_FEE_PCT}/yıl
  • Performans: %{self.PERFORMANCE_FEE_PCT}

🔐 TRUSTLESS:
  • ERC-4626 Uyumlu
  • Akıllı Kontratta

</tokenized_vault>
"""
        return report


class VaultStrategyInterface:
    """
    Vault-Strategy Interface.
    
    Python trading bot ile akıllı kontrat arasındaki köprü.
    """
    
    def __init__(self, vault: TokenizedVault):
        self.vault = vault
        self.pending_trades = []
        self.executed_trades = []
    
    def get_available_capital(self) -> float:
        """Kullanılabilir sermaye."""
        return self.vault.total_assets
    
    def report_pnl(self, pnl: float):
        """PnL bildirimi."""
        new_total = self.vault.total_assets + pnl
        self.vault.update_nav(new_total)
        
        print(f"{Fore.CYAN}📊 Vault NAV güncellendi: PnL ${pnl:+.2f}{Style.RESET_ALL}", flush=True)
    
    def submit_trade(self, trade: Dict) -> Dict:
        """İşlem gönder."""
        trade["vault_id"] = self.vault.vault_name
        trade["timestamp"] = datetime.now().isoformat()
        trade["status"] = "PENDING"
        
        self.pending_trades.append(trade)
        
        return trade
    
    def confirm_trade(self, trade_id: str, execution_price: float, pnl: float):
        """İşlem onayı."""
        for trade in self.pending_trades:
            if trade.get("id") == trade_id:
                trade["status"] = "EXECUTED"
                trade["execution_price"] = execution_price
                trade["pnl"] = pnl
                
                self.executed_trades.append(trade)
                self.pending_trades.remove(trade)
                
                # NAV güncelle
                self.report_pnl(pnl)
                
                break
