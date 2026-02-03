"""
Backtest Learning and Pattern Analysis
Author: Erdinc Erdogan
Purpose: Analyzes historical backtest results to identify winning/losing patterns and generate actionable trading insights.
References:
- Trade Pattern Recognition
- Win Rate Analysis
- Performance Attribution
Usage:
    learner = BacktestLearning(memory=memory_store)
    analysis = learner.analyze_patterns(trades)
    insights = learner.generate_insights(analysis)
"""
from typing import List, Dict


class BacktestLearning:
    """
    Backtest sonuçlarından ders çıkarma.
    """
    
    def __init__(self, memory=None):
        self.memory = memory
    
    def analyze_patterns(self, trades: List[Dict]) -> Dict:
        """Trade pattern'larını analiz et."""
        if not trades:
            return {}
        
        wins = [t for t in trades if t.get('outcome') == 'WIN']
        losses = [t for t in trades if t.get('outcome') == 'LOSS']
        
        return {
            'total_trades': len(trades),
            'win_rate': len(wins) / len(trades) * 100 if trades else 0,
            'avg_win': sum(t.get('profit_pct', 0) for t in wins) / len(wins) if wins else 0,
            'avg_loss': sum(t.get('profit_pct', 0) for t in losses) / len(losses) if losses else 0
        }
    
    def generate_insights(self, analysis: Dict) -> str:
        """Insights oluştur."""
        if not analysis:
            return ""
        
        insights = []
        
        win_rate = analysis.get('win_rate', 0)
        if win_rate > 60:
            insights.append("✅ Yüksek kazanma oranı - Strateji çalışıyor")
        elif win_rate < 40:
            insights.append("⚠️ Düşük kazanma oranı - Strateji gözden geçirilmeli")
        
        return "\n".join(insights)
