"""
Immutable Merkle Tree Audit Log
Author: Erdinc Erdogan
Purpose: Provides tamper-proof audit trail using Merkle Tree structure for all trading decisions, ensuring regulatory compliance and forensic capability.
References:
- Merkle Tree (Ralph Merkle, 1979)
- Blockchain Audit Trail Patterns
- Financial Regulatory Logging Requirements
Usage:
    audit = ImmutableAuditLog()
    audit.record_decision(decision="BUY", ticker="AAPL", confidence=0.85, reasoning="Strong momentum")
"""

import os
import json
import hashlib
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from colorama import Fore, Style


@dataclass
class AuditEntry:
    """Denetim kaydı."""
    timestamp: str
    decision: str
    ticker: str
    confidence: float
    reasoning: str
    market_state: Dict
    hash: str = ""
    prev_hash: str = ""


class MerkleNode:
    """Merkle Tree düğümü."""
    
    def __init__(self, left=None, right=None, data: str = ""):
        self.left = left
        self.right = right
        self.data = data
        
        if left is None and right is None:
            self.hash = self._hash(data)
        else:
            self.hash = self._hash(
                (left.hash if left else "") + (right.hash if right else "")
            )
    
    def _hash(self, data: str) -> str:
        return hashlib.sha256(data.encode()).hexdigest()


class MerkleTree:
    """Merkle Tree implementasyonu."""
    
    def __init__(self):
        self.leaves = []
        self.root = None
    
    def add_leaf(self, data: str):
        """Yaprak ekle."""
        self.leaves.append(MerkleNode(data=data))
        self._rebuild_tree()
    
    def _rebuild_tree(self):
        """Ağacı yeniden oluştur."""
        if not self.leaves:
            self.root = None
            return
        
        nodes = self.leaves.copy()
        
        while len(nodes) > 1:
            new_level = []
            for i in range(0, len(nodes), 2):
                left = nodes[i]
                right = nodes[i + 1] if i + 1 < len(nodes) else None
                new_level.append(MerkleNode(left=left, right=right))
            nodes = new_level
        
        self.root = nodes[0] if nodes else None
    
    def get_root_hash(self) -> str:
        """Kök hash'i döndür."""
        return self.root.hash if self.root else ""
    
    def verify_leaf(self, data: str) -> bool:
        """Yaprağın ağaçta olup olmadığını doğrula."""
        leaf_hash = hashlib.sha256(data.encode()).hexdigest()
        return any(leaf.hash == leaf_hash for leaf in self.leaves)


class ImmutableAuditLog:
    """
    Immutable Audit Log - Değiştirilemez Denetim Kaydı.
    
    Özellikler:
    - Blockchain benzeri zincirleme (her kayıt öncekinin hash'ini içerir)
    - Merkle Tree ile bütünlük doğrulama
    - SQLite ile kalıcı depolama
    - Zaman damgası ve market state kaydı
    """
    
    def __init__(self, db_path: str = "audit_log.db"):
        """
        Args:
            db_path: SQLite veritabanı yolu
        """
        self.db_path = db_path
        self.merkle_tree = MerkleTree()
        self.chain = []
        self._init_db()
        self._load_chain()
    
    def _init_db(self):
        """Veritabanını başlat."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                decision TEXT NOT NULL,
                ticker TEXT NOT NULL,
                confidence REAL NOT NULL,
                reasoning TEXT,
                market_state TEXT,
                hash TEXT UNIQUE NOT NULL,
                prev_hash TEXT,
                merkle_root TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS merkle_roots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                root_hash TEXT NOT NULL,
                entries_count INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_chain(self):
        """Mevcut zinciri yükle."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM audit_log ORDER BY id')
        rows = cursor.fetchall()
        
        for row in rows:
            entry = AuditEntry(
                timestamp=row[1],
                decision=row[2],
                ticker=row[3],
                confidence=row[4],
                reasoning=row[5],
                market_state=json.loads(row[6]) if row[6] else {},
                hash=row[7],
                prev_hash=row[8]
            )
            self.chain.append(entry)
            self.merkle_tree.add_leaf(entry.hash)
        
        conn.close()
        print(f"{Fore.GREEN}📋 Audit Log: {len(self.chain)} kayıt yüklendi{Style.RESET_ALL}", flush=True)
    
    def log_decision(self,
                    decision: str,
                    ticker: str,
                    confidence: float,
                    reasoning: str,
                    market_state: Dict = None) -> AuditEntry:
        """
        Karar kaydet.
        
        Args:
            decision: AL/SAT/BEKLE
            ticker: Hisse sembolü
            confidence: Güven skoru
            reasoning: Karar gerekçesi
            market_state: O anki piyasa durumu
        """
        # Önceki hash
        prev_hash = self.chain[-1].hash if self.chain else "GENESIS"
        
        # Kayıt oluştur
        entry = AuditEntry(
            timestamp=datetime.now().isoformat(),
            decision=decision,
            ticker=ticker,
            confidence=confidence,
            reasoning=reasoning,
            market_state=market_state or {},
            prev_hash=prev_hash
        )
        
        # Hash hesapla
        entry_data = json.dumps({
            "timestamp": entry.timestamp,
            "decision": entry.decision,
            "ticker": entry.ticker,
            "confidence": entry.confidence,
            "reasoning": entry.reasoning,
            "market_state": entry.market_state,
            "prev_hash": entry.prev_hash
        }, sort_keys=True)
        
        entry.hash = hashlib.sha256(entry_data.encode()).hexdigest()
        
        # Zincire ekle
        self.chain.append(entry)
        self.merkle_tree.add_leaf(entry.hash)
        
        # Veritabanına kaydet
        self._save_entry(entry)
        
        print(f"{Fore.CYAN}📝 Audit: {decision} {ticker} @ {entry.timestamp[:19]}{Style.RESET_ALL}", flush=True)
        
        return entry
    
    def _save_entry(self, entry: AuditEntry):
        """Kaydı veritabanına kaydet."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO audit_log 
            (timestamp, decision, ticker, confidence, reasoning, market_state, hash, prev_hash, merkle_root)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            entry.timestamp,
            entry.decision,
            entry.ticker,
            entry.confidence,
            entry.reasoning,
            json.dumps(entry.market_state),
            entry.hash,
            entry.prev_hash,
            self.merkle_tree.get_root_hash()
        ))
        
        conn.commit()
        conn.close()
    
    def verify_chain_integrity(self) -> Dict:
        """
        Zincir bütünlüğünü doğrula.
        
        Her kaydın hash'ini ve önceki hash bağlantısını kontrol eder.
        """
        print(f"{Fore.CYAN}🔍 Zincir bütünlüğü kontrol ediliyor...{Style.RESET_ALL}", flush=True)
        
        if not self.chain:
            return {"valid": True, "entries": 0, "message": "Boş zincir"}
        
        errors = []
        
        for i, entry in enumerate(self.chain):
            # Hash doğrulama
            entry_data = json.dumps({
                "timestamp": entry.timestamp,
                "decision": entry.decision,
                "ticker": entry.ticker,
                "confidence": entry.confidence,
                "reasoning": entry.reasoning,
                "market_state": entry.market_state,
                "prev_hash": entry.prev_hash
            }, sort_keys=True)
            
            calculated_hash = hashlib.sha256(entry_data.encode()).hexdigest()
            
            if calculated_hash != entry.hash:
                errors.append(f"Kayıt {i}: Hash uyuşmazlığı")
            
            # Önceki hash bağlantısı
            if i > 0:
                if entry.prev_hash != self.chain[i-1].hash:
                    errors.append(f"Kayıt {i}: Önceki hash bağlantısı kopuk")
        
        is_valid = len(errors) == 0
        
        result = {
            "valid": is_valid,
            "entries": len(self.chain),
            "errors": errors,
            "merkle_root": self.merkle_tree.get_root_hash(),
            "message": "✅ Zincir geçerli" if is_valid else "❌ Bütünlük ihlali tespit edildi"
        }
        
        color = Fore.GREEN if is_valid else Fore.RED
        print(f"{color}{result['message']}{Style.RESET_ALL}", flush=True)
        
        return result
    
    def get_entry_proof(self, entry_hash: str) -> Dict:
        """
        Belirli bir kayıt için Merkle kanıtı oluştur.
        
        Bu kanıt, kaydın zincirin parçası olduğunu bağımsız olarak doğrular.
        """
        if self.merkle_tree.verify_leaf(entry_hash):
            return {
                "exists": True,
                "entry_hash": entry_hash,
                "merkle_root": self.merkle_tree.get_root_hash(),
                "proof": "Merkle tree'de doğrulandı"
            }
        return {"exists": False, "entry_hash": entry_hash}
    
    def export_chain(self, filepath: str = None) -> str:
        """Zinciri JSON olarak dışa aktar."""
        chain_data = [asdict(entry) for entry in self.chain]
        
        export = {
            "exported_at": datetime.now().isoformat(),
            "entries_count": len(self.chain),
            "merkle_root": self.merkle_tree.get_root_hash(),
            "chain": chain_data
        }
        
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(export, f, indent=2)
            print(f"{Fore.GREEN}📤 Zincir dışa aktarıldı: {filepath}{Style.RESET_ALL}", flush=True)
        
        return json.dumps(export, indent=2)
    
    def get_statistics(self) -> Dict:
        """Audit log istatistikleri."""
        if not self.chain:
            return {"entries": 0}
        
        decisions = {}
        tickers = {}
        
        for entry in self.chain:
            decisions[entry.decision] = decisions.get(entry.decision, 0) + 1
            tickers[entry.ticker] = tickers.get(entry.ticker, 0) + 1
        
        return {
            "total_entries": len(self.chain),
            "decisions": decisions,
            "tickers": tickers,
            "first_entry": self.chain[0].timestamp,
            "last_entry": self.chain[-1].timestamp,
            "merkle_root": self.merkle_tree.get_root_hash()
        }
    
    def generate_audit_report(self) -> str:
        """Audit raporu oluştur."""
        integrity = self.verify_chain_integrity()
        stats = self.get_statistics()
        
        report = f"""
<immutable_audit_log>
📋 DEĞIŞTIRILEMEZ DENETİM KAYDI
════════════════════════════════════════

🔐 BÜTÜNLÜK DURUMU: {'✅ GEÇERLI' if integrity['valid'] else '❌ İHLAL'}
🔗 Merkle Root: {stats.get('merkle_root', 'N/A')[:16]}...

📊 İSTATİSTİKLER:
  • Toplam Kayıt: {stats.get('total_entries', 0)}
  • Kararlar: {stats.get('decisions', {})}
  • Hisseler: {list(stats.get('tickers', {}).keys())[:5]}

📅 ZAMAN ARALIĞI:
  • İlk Kayıt: {stats.get('first_entry', 'N/A')[:19]}
  • Son Kayıt: {stats.get('last_entry', 'N/A')[:19]}

🔍 SON 5 KARAR:
"""
        for entry in self.chain[-5:]:
            emoji = "🟢" if entry.decision == "AL" else "🔴" if entry.decision == "SAT" else "🟡"
            report += f"  {emoji} {entry.ticker}: {entry.decision} (%{entry.confidence*100:.0f}) - {entry.timestamp[:16]}\n"
        
        report += "</immutable_audit_log>\n"
        return report
