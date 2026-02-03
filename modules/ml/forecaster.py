"""
LSTM/GRU Time-Series Price Forecaster
Author: Erdinc Erdogan
Purpose: Provides pure mathematical price forecasting using LSTM/GRU neural networks to complement text-based RAG analysis.
References:
- Long Short-Term Memory (Hochreiter & Schmidhuber, 1997)
- Gated Recurrent Units (Cho et al., 2014)
- Sequence-to-Sequence Forecasting
Usage:
    forecaster = TimeSeriesForecaster(model_type="lstm", lookback=60)
    forecaster.build_model(input_shape=(60, 1))
    prediction = forecaster.predict(prices)
"""
import os
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from colorama import Fore, Style


class TimeSeriesForecaster:
    """
    LSTM/GRU ile zaman serisi tahmini.
    
    RAG metin analizi yaparken, paralel çalışan bu model
    saf matematiksel fiyat tahmini yapar.
    """
    
    def __init__(self, model_type: str = "lstm", lookback: int = 60):
        """
        Args:
            model_type: "lstm" veya "gru"
            lookback: Kaç geçmiş veri noktası kullanılacak
        """
        self.model_type = model_type
        self.lookback = lookback
        self.model = None
        self.scaler = None
        self.is_trained = False
        
        # TensorFlow lazy import
        self.tf = None
        self._import_tensorflow()
    
    def _import_tensorflow(self):
        """TensorFlow'u lazy import et."""
        try:
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')
            self.tf = tf
            print(f"{Fore.GREEN}  → TensorFlow {tf.__version__} yüklendi{Style.RESET_ALL}", flush=True)
        except ImportError:
            print(f"{Fore.YELLOW}  → TensorFlow yüklü değil, demo mode{Style.RESET_ALL}", flush=True)
            self.tf = None
    
    def build_model(self, input_shape: Tuple[int, int], units: int = 64) -> None:
        """
        LSTM/GRU modeli oluştur.
        
        Args:
            input_shape: (timesteps, features)
            units: LSTM/GRU unit sayısı
        """
        if self.tf is None:
            return
        
        tf = self.tf
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
        ])
        
        if self.model_type == "lstm":
            model.add(tf.keras.layers.LSTM(units, return_sequences=True))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.LSTM(units // 2))
        else:  # GRU
            model.add(tf.keras.layers.GRU(units, return_sequences=True))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.GRU(units // 2))
        
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(1))
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.model = model
        
        print(f"{Fore.GREEN}  → {self.model_type.upper()} model oluşturuldu{Style.RESET_ALL}", flush=True)
    
    def prepare_data(self, prices: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Veriyi LSTM formatına çevir.
        
        Args:
            prices: Fiyat listesi
            
        Returns:
            X: (samples, timesteps, features)
            y: (samples,)
        """
        from sklearn.preprocessing import MinMaxScaler
        
        data = np.array(prices).reshape(-1, 1)
        
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i - self.lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        X = np.array(X)
        y = np.array(y)
        
        # LSTM için reshape: (samples, timesteps, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        return X, y
    
    def train(self, prices: List[float], epochs: int = 50, batch_size: int = 32) -> Dict:
        """
        Modeli eğit.
        
        Args:
            prices: Geçmiş fiyat listesi
            epochs: Eğitim epoch sayısı
            batch_size: Batch boyutu
        """
        if self.tf is None:
            return {"error": "TensorFlow yüklü değil"}
        
        print(f"{Fore.CYAN}  → LSTM eğitimi başlıyor ({len(prices)} veri)...{Style.RESET_ALL}", flush=True)
        
        try:
            X, y = self.prepare_data(prices)
            
            # Train/test split
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            if self.model is None:
                self.build_model(input_shape=(X.shape[1], X.shape[2]))
            
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                verbose=0
            )
            
            self.is_trained = True
            
            # Performans metrikleri
            train_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            
            print(f"{Fore.GREEN}  → Eğitim tamamlandı: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}{Style.RESET_ALL}", flush=True)
            
            return {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epochs": epochs,
                "samples": len(X),
                "status": "success"
            }
            
        except Exception as e:
            print(f"{Fore.RED}  → Eğitim hatası: {e}{Style.RESET_ALL}", flush=True)
            return {"error": str(e)}
    
    def predict(self, prices: List[float], days_ahead: int = 5) -> Dict:
        """
        Gelecek fiyatları tahmin et.
        
        Args:
            prices: Son fiyatlar (en az lookback kadar)
            days_ahead: Kaç gün ileri tahmin
        """
        # TensorFlow yoksa veya model eğitilmemişse demo tahmin
        if self.tf is None or not self.is_trained:
            return self._demo_predict(prices, days_ahead)
        
        try:
            # Son lookback kadar veriyi al
            recent = np.array(prices[-self.lookback:]).reshape(-1, 1)
            scaled = self.scaler.transform(recent)
            
            predictions = []
            current_input = scaled.reshape(1, self.lookback, 1)
            
            for _ in range(days_ahead):
                pred = self.model.predict(current_input, verbose=0)[0, 0]
                predictions.append(pred)
                
                # Sliding window
                current_input = np.roll(current_input, -1, axis=1)
                current_input[0, -1, 0] = pred
            
            # Inverse transform
            predictions = self.scaler.inverse_transform(
                np.array(predictions).reshape(-1, 1)
            ).flatten()
            
            current_price = prices[-1]
            
            return {
                "current_price": current_price,
                "predictions": predictions.tolist(),
                "days_ahead": days_ahead,
                "direction": "UP" if predictions[-1] > current_price else "DOWN",
                "change_pct": ((predictions[-1] - current_price) / current_price) * 100,
                "model": self.model_type.upper(),
                "confidence": self._calculate_confidence(predictions, current_price)
            }
            
        except Exception as e:
            return self._demo_predict(prices, days_ahead)
    
    def _demo_predict(self, prices: List[float], days_ahead: int) -> Dict:
        """Demo tahmin (TensorFlow olmadan)."""
        current_price = prices[-1] if prices else 100
        
        # Basit momentum bazlı tahmin
        if len(prices) >= 5:
            momentum = (prices[-1] - prices[-5]) / prices[-5]
        else:
            momentum = 0.01
        
        predictions = []
        price = current_price
        for i in range(days_ahead):
            # Momentum + rastgele noise
            change = momentum * 0.5 + np.random.uniform(-0.02, 0.02)
            price = price * (1 + change)
            predictions.append(price)
        
        return {
            "current_price": current_price,
            "predictions": predictions,
            "days_ahead": days_ahead,
            "direction": "UP" if predictions[-1] > current_price else "DOWN",
            "change_pct": ((predictions[-1] - current_price) / current_price) * 100,
            "model": "MOMENTUM (Demo)",
            "confidence": 0.5
        }
    
    def _calculate_confidence(self, predictions: np.ndarray, current: float) -> float:
        """Tahmin güvenilirliğini hesapla."""
        # Volatilite bazlı güven
        if len(predictions) < 2:
            return 0.5
        
        volatility = np.std(predictions) / current
        confidence = max(0.3, min(0.9, 1 - volatility * 10))
        return confidence
    
    def generate_forecast_report(self, ticker: str, prices: List[float], days: int = 5) -> str:
        """Tahmin raporu oluştur."""
        forecast = self.predict(prices, days)
        
        direction_emoji = "📈" if forecast["direction"] == "UP" else "📉"
        
        report = f"""
<time_series_forecast>
🔮 ZAMAN SERİSİ TAHMİNİ - {ticker}
════════════════════════════════════════
Model: {forecast['model']}
Mevcut Fiyat: ${forecast['current_price']:.2f}

{direction_emoji} {days} GÜNLÜK TAHMİN:
"""
        for i, pred in enumerate(forecast["predictions"], 1):
            change = ((pred - forecast['current_price']) / forecast['current_price']) * 100
            report += f"  Gün {i}: ${pred:.2f} ({change:+.2f}%)\n"
        
        report += f"""
📊 ÖZET:
  • Yön: {forecast['direction']}
  • Beklenen Değişim: {forecast['change_pct']:+.2f}%
  • Güven: %{forecast['confidence']*100:.0f}

</time_series_forecast>
"""
        return report
