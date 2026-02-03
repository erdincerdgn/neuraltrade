"""
Secure Python REPL for Financial Calculations
Author: Erdinc Erdogan
Purpose: Provides sandboxed Python execution for risk/reward calculations, position sizing, and financial computations with security restrictions.
References:
- Sandboxed Code Execution
- Risk/Reward Calculation
- Position Sizing Formulas
Usage:
    repl = PythonREPL()
    result = repl.calculate_risk_reward(entry=100, stop=95, target=110)
"""
import traceback
from io import StringIO
from contextlib import redirect_stdout
from colorama import Fore, Style


class PythonREPL:
    """Python REPL ile finansal hesaplama."""
    
    SAFE_FUNCTIONS = {'abs', 'round', 'min', 'max', 'sum', 'len', 'float', 'int', 'str', 'range'}
    
    def __init__(self):
        self.globals_dict = {"__builtins__": {k: __builtins__[k] for k in self.SAFE_FUNCTIONS if k in dir(__builtins__)}}
        try:
            import numpy as np
            import pandas as pd
            self.globals_dict.update({"np": np, "pd": pd})
        except:
            pass
    
    def execute(self, code: str) -> str:
        """Güvenli kod çalıştır."""
        dangerous = ['import os', 'import sys', 'eval(', 'exec(', '__import__', 'subprocess']
        if any(d in code.lower() for d in dangerous):
            return "⚠️ Güvenlik: Bu kod çalıştırılamaz"
        
        try:
            stdout = StringIO()
            with redirect_stdout(stdout):
                exec(code, self.globals_dict)
            return stdout.getvalue().strip() or "✅ Çalıştırıldı"
        except Exception as e:
            return f"❌ Hata: {str(e)}"
    
    def calculate_risk_reward(self, entry: float, stop: float, target: float) -> str:
        risk = abs(entry - stop)
        reward = abs(target - entry)
        ratio = reward / risk if risk > 0 else 0
        return f"Risk/Reward: 1:{ratio:.2f} (Entry: {entry}, SL: {stop}, TP: {target})"
    
    def calculate_position_size(self, capital: float, risk_pct: float, stop_pips: float) -> str:
        risk_amount = capital * (risk_pct / 100)
        position = risk_amount / stop_pips if stop_pips > 0 else 0
        return f"Pozisyon: {position:.2f} lot (Sermaye: ${capital}, Risk: %{risk_pct}, SL: {stop_pips} pip)"
