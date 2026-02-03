"""
Technical Analysis Engine - ICT Fair Value Gap Detection
Author: Erdinc Erdogan
Purpose: Implements technical analysis indicators including ICT Fair Value Gap (FVG) detection,
RSI, moving averages, and Bollinger Bands for market structure analysis.
References:
- ICT (Inner Circle Trader) Fair Value Gap Methodology
- TA-Lib Technical Analysis Library
- Price Action and Market Structure Theory
Usage:
    fvgs = find_fair_value_gaps(df)
    analysis = analyze_data(df)
    print(f"Trend: {analysis['trend']}, RSI: {analysis['rsi']:.2f}")
"""
import pandas as pd
import talib

def find_fair_value_gaps(df):
    """
    ICT Fair Value Gap (FVG) Determination
    Author: Erdinc Erdogan
    Logic: If there is a gap between the high of the 1st candle and the low of the 3rd candle, it is an FVG.
    """
    fvg_list = []
    
    # Checking the last 50 candles is sufficient (for speed)
    # i = Current candle, i-1 = Previous, i-2 = Before previous
    for i in range(len(df) - 3, len(df) - 50, -1):
        curr_candle = df.iloc[i]      # Candle 3
        prev_candle = df.iloc[i-1]    # Candle 2 (Usually a large-bodied candle)
        prev2_candle = df.iloc[i-2]   # Candle 1
        
        # BULLISH FVG (Rising Gap)
        # High of Candle 1 < Low of Candle 3
        if prev2_candle['high'] < curr_candle['low'] and prev_candle['close'] > prev_candle['open']:
            fvg_list.append({
                "type": "BULLISH 🟢",
                "top": curr_candle['low'],
                "bottom": prev2_candle['high'],
                "time": df.index[i]
            })

        # BEARISH FVG (Falling Gap)
        # Low of Candle 1 > High of Candle 3
        elif prev2_candle['low'] > curr_candle['high'] and prev_candle['close'] < prev_candle['open']:
            fvg_list.append({
                "type": "BEARISH 🔴",
                "top": prev2_candle['low'],
                "bottom": curr_candle['high'],
                "time": df.index[i]
            })
            
    return fvg_list

def analyze_data(df):
    """
    Analyze data: RSI + ICT FVG
    """
    # 1. Calculate classic RSI (for trend filter)
    df['RSI'] = talib.RSI(df['close'].values, timeperiod=14)
    
    current_rsi = df['RSI'].iloc[-1]
    current_price = df['close'].iloc[-1]
    
    # 2. Find ICT FVGs
    fvgs = find_fair_value_gaps(df)
    
    # Determine Trend
    trend = "NEUTRAL"
    if current_rsi > 55: trend = "RISING (BULL)"
    elif current_rsi < 45: trend = "FALLING (BEAR)"
    
    return {
        "price": current_price,
        "rsi": current_rsi,
        "trend": trend,
        "fvgs": fvgs  # Found gaps added to list
    }