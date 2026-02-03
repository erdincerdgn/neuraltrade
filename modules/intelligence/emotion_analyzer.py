"""
Emotion Analyzer for Earnings Calls
Author: Erdinc Erdogan
Purpose: Analyzes CEO vocal stress indicators, linguistic patterns, and micro-expressions during earnings calls to detect hidden sentiment signals.
References:
- OpenAI Whisper (Speech-to-Text)
- Librosa (Audio Prosody Analysis)
- Micro-Expression Analysis (Ekman)
Usage:
    analyzer = EmotionAnalyzer()
    result = analyzer.analyze_earnings_call(audio_path="call.mp3", ticker="AAPL")
"""
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from colorama import Fore, Style


class EmotionAnalyzer:
    """
    Emotion Analyzer v2.
    
    CEO ve yöneticilerin earnings call kayıtlarından
    duygusal durumu analiz eder:
    - Ses tonu analizi (Whisper + prosody)
    - Konuşma hızı ve duraklamalar
    - Mikro-ifadeler (video varsa)
    - Kelime seçimi ve sentiment
    """
    
    # Stres göstergeleri
    STRESS_INDICATORS = {
        "vocal": {
            "pitch_increase": "Ses tonu yükseldi - stres belirtisi",
            "speech_rate_increase": "Konuşma hızı arttı - kaçınma",
            "filler_words": "Dolgu kelimeler arttı (um, uh) - belirsizlik",
            "long_pauses": "Uzun duraklamalar - düşünme/kaçınma",
            "voice_tremor": "Ses titremesi - yoğun stres"
        },
        "linguistic": {
            "hedging": "Kaçak ifadeler (belki, muhtemelen) - belirsizlik",
            "passive_voice": "Pasif yapı kullanımı - sorumluluk kaçınma",
            "future_tense_decrease": "Gelecek zaman kullanımı azaldı - vizyon yok",
            "negative_sentiment": "Negatif kelime oranı arttı"
        },
        "micro_expressions": {
            "contempt": "Küçümseme ifadesi - gizli olumsuzluk",
            "fear": "Korku ifadesi - endişe",
            "surprise": "Şaşkınlık - beklenmeyen soru",
            "disgust": "Tiksinme - konudan rahatsızlık"
        }
    }
    
    def __init__(self):
        self.whisper_available = False
        self.librosa_available = False
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Bağımlılıkları kontrol et."""
        try:
            import whisper
            self.whisper_available = True
        except ImportError:
            print(f"{Fore.YELLOW}⚠️ Whisper yüklü değil, ses analizi simüle edilecek{Style.RESET_ALL}", flush=True)
        
        try:
            import librosa
            self.librosa_available = True
        except ImportError:
            print(f"{Fore.YELLOW}⚠️ Librosa yüklü değil, prosody analizi simüle edilecek{Style.RESET_ALL}", flush=True)
    
    def analyze_earnings_call(self, 
                             audio_path: str = None,
                             transcript: str = None,
                             ticker: str = None) -> Dict:
        """
        Earnings call analizi.
        
        Args:
            audio_path: Ses dosyası yolu
            transcript: Metin transkripti
            ticker: Şirket sembolü
        """
        print(f"{Fore.CYAN}🎭 Earnings Call emotion analizi: {ticker}...{Style.RESET_ALL}", flush=True)
        
        results = {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "vocal_analysis": None,
            "linguistic_analysis": None,
            "overall_sentiment": None,
            "stress_level": None,
            "trading_signal": None
        }
        
        # Ses analizi
        if audio_path:
            results["vocal_analysis"] = self._analyze_audio(audio_path)
        
        # Metin analizi
        if transcript:
            results["linguistic_analysis"] = self._analyze_transcript(transcript)
        else:
            # Demo transkript
            results["linguistic_analysis"] = self._analyze_transcript(self._demo_transcript())
        
        # Birleşik değerlendirme
        results["overall_sentiment"] = self._calculate_overall_sentiment(results)
        results["stress_level"] = self._calculate_stress_level(results)
        results["trading_signal"] = self._generate_trading_signal(results)
        
        return results
    
    def _analyze_audio(self, audio_path: str) -> Dict:
        """Ses dosyasını analiz et."""
        if self.librosa_available and os.path.exists(audio_path):
            return self._real_audio_analysis(audio_path)
        else:
            return self._simulated_audio_analysis()
    
    def _real_audio_analysis(self, audio_path: str) -> Dict:
        """Gerçek ses analizi (librosa ile)."""
        import librosa
        
        y, sr = librosa.load(audio_path)
        
        # Pitch (F0) analizi
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        avg_pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 150
        
        # Tempo / konuşma hızı
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Enerji / ses yüksekliği
        rms = librosa.feature.rms(y=y)
        avg_energy = np.mean(rms)
        
        # Spectral features (ses kalitesi)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        
        return {
            "avg_pitch_hz": float(avg_pitch),
            "pitch_variance": float(np.std(pitches[pitches > 0])) if np.any(pitches > 0) else 0,
            "tempo_bpm": float(tempo),
            "avg_energy": float(avg_energy),
            "spectral_centroid": float(np.mean(spectral_centroid)),
            "analysis_type": "REAL"
        }
    
    def _simulated_audio_analysis(self) -> Dict:
        """Simüle edilmiş ses analizi."""
        # Baseline değerler (normal CEO konuşması)
        baseline = {
            "avg_pitch_hz": 150,
            "tempo_bpm": 120,  # kelime/dakika
            "avg_energy": 0.1
        }
        
        # Rastgele stres seviyesi
        stress_factor = np.random.uniform(0.8, 1.4)
        
        return {
            "avg_pitch_hz": baseline["avg_pitch_hz"] * stress_factor,
            "pitch_variance": 20 * stress_factor,
            "tempo_bpm": baseline["tempo_bpm"] * stress_factor,
            "avg_energy": baseline["avg_energy"] * stress_factor,
            "stress_factor": stress_factor,
            "analysis_type": "SIMULATED",
            "indicators": self._detect_vocal_stress(stress_factor)
        }
    
    def _detect_vocal_stress(self, stress_factor: float) -> List[str]:
        """Vokal stres göstergelerini tespit et."""
        indicators = []
        
        if stress_factor > 1.2:
            indicators.append("pitch_increase")
            indicators.append("speech_rate_increase")
        if stress_factor > 1.3:
            indicators.append("voice_tremor")
        
        return indicators
    
    def _analyze_transcript(self, transcript: str) -> Dict:
        """Metin analizi."""
        words = transcript.lower().split()
        total_words = len(words)
        
        # Filler words (dolgu kelimeler)
        filler_words = ["um", "uh", "like", "you know", "basically", "actually", "so"]
        filler_count = sum(1 for w in words if w in filler_words)
        filler_ratio = filler_count / total_words if total_words > 0 else 0
        
        # Hedging (kaçak ifadeler)
        hedging_words = ["maybe", "perhaps", "might", "could", "possibly", "somewhat", 
                        "relatively", "fairly", "belki", "muhtemelen", "sanırım"]
        hedging_count = sum(1 for w in words if w in hedging_words)
        hedging_ratio = hedging_count / total_words if total_words > 0 else 0
        
        # Negatif kelimeler
        negative_words = ["decline", "challenge", "difficult", "concern", "issue", 
                         "problem", "risk", "uncertainty", "pressure", "headwind",
                         "düşüş", "zorluk", "endişe", "risk"]
        negative_count = sum(1 for w in words if w in negative_words)
        negative_ratio = negative_count / total_words if total_words > 0 else 0
        
        # Pozitif kelimeler
        positive_words = ["growth", "strong", "excellent", "improvement", "opportunity",
                         "success", "confident", "optimistic", "momentum", "record",
                         "büyüme", "güçlü", "başarı", "fırsat"]
        positive_count = sum(1 for w in words if w in positive_words)
        positive_ratio = positive_count / total_words if total_words > 0 else 0
        
        # Net sentiment
        sentiment_score = positive_ratio - negative_ratio
        
        # Gelecek zaman kullanımı
        future_words = ["will", "expect", "anticipate", "forecast", "project", "plan"]
        future_count = sum(1 for w in words if w in future_words)
        future_ratio = future_count / total_words if total_words > 0 else 0
        
        # Stres göstergeleri
        indicators = []
        if filler_ratio > 0.02:
            indicators.append("filler_words")
        if hedging_ratio > 0.01:
            indicators.append("hedging")
        if negative_ratio > 0.015:
            indicators.append("negative_sentiment")
        if future_ratio < 0.005:
            indicators.append("future_tense_decrease")
        
        return {
            "total_words": total_words,
            "filler_ratio": filler_ratio,
            "hedging_ratio": hedging_ratio,
            "negative_ratio": negative_ratio,
            "positive_ratio": positive_ratio,
            "sentiment_score": sentiment_score,
            "future_ratio": future_ratio,
            "indicators": indicators
        }
    
    def _demo_transcript(self) -> str:
        """Demo transkript."""
        return """
        Thank you for joining our quarterly earnings call. 
        We are pleased to report strong performance across all segments.
        Revenue grew by 15% year over year, exceeding our expectations.
        However, we face some challenges in the supply chain.
        Um, the macroeconomic environment remains somewhat uncertain.
        We believe, uh, that we are well positioned for the coming quarters.
        Our focus on innovation will, perhaps, drive continued growth.
        We expect to see improvement in margins going forward.
        """
    
    def _calculate_overall_sentiment(self, results: Dict) -> Dict:
        """Genel sentiment hesapla."""
        linguistic = results.get("linguistic_analysis", {})
        vocal = results.get("vocal_analysis", {})
        
        # Metin sentiment skoru
        text_score = linguistic.get("sentiment_score", 0)
        
        # Ses stres faktörü
        vocal_stress = vocal.get("stress_factor", 1.0) if vocal else 1.0
        
        # Birleşik skor
        combined_score = text_score - (vocal_stress - 1) * 0.5
        
        if combined_score > 0.005:
            sentiment = "POSITIVE"
        elif combined_score < -0.005:
            sentiment = "NEGATIVE"
        else:
            sentiment = "NEUTRAL"
        
        return {
            "text_score": text_score,
            "vocal_adjustment": -(vocal_stress - 1) * 0.5,
            "combined_score": combined_score,
            "sentiment": sentiment
        }
    
    def _calculate_stress_level(self, results: Dict) -> Dict:
        """Stres seviyesi hesapla."""
        all_indicators = []
        
        linguistic = results.get("linguistic_analysis", {})
        vocal = results.get("vocal_analysis", {})
        
        all_indicators.extend(linguistic.get("indicators", []))
        all_indicators.extend(vocal.get("indicators", []))
        
        num_indicators = len(all_indicators)
        
        if num_indicators >= 4:
            level = "HIGH"
            score = 0.8
        elif num_indicators >= 2:
            level = "MEDIUM"
            score = 0.5
        else:
            level = "LOW"
            score = 0.2
        
        return {
            "level": level,
            "score": score,
            "indicators_found": all_indicators,
            "indicator_count": num_indicators
        }
    
    def _generate_trading_signal(self, results: Dict) -> Dict:
        """Trading sinyali üret."""
        sentiment = results.get("overall_sentiment", {}).get("sentiment", "NEUTRAL")
        stress_level = results.get("stress_level", {}).get("level", "LOW")
        
        # Karar matrisi
        decision_matrix = {
            ("POSITIVE", "LOW"): ("BUY", 0.8, "CEO pozitif ve rahat"),
            ("POSITIVE", "MEDIUM"): ("BUY", 0.6, "Pozitif ama hafif stres var"),
            ("POSITIVE", "HIGH"): ("HOLD", 0.5, "Pozitif söylem ama yüksek stres - dikkat"),
            ("NEUTRAL", "LOW"): ("HOLD", 0.5, "Nötr duruş"),
            ("NEUTRAL", "MEDIUM"): ("HOLD", 0.4, "Nötr ama stres belirtileri"),
            ("NEUTRAL", "HIGH"): ("SELL", 0.6, "Nötr görünümlü ama stresli - kötü haber gelebilir"),
            ("NEGATIVE", "LOW"): ("SELL", 0.6, "Negatif içerik"),
            ("NEGATIVE", "MEDIUM"): ("SELL", 0.7, "Negatif ve stresli"),
            ("NEGATIVE", "HIGH"): ("STRONG_SELL", 0.9, "Yüksek stresle negatif - ciddi sorun olabilir"),
        }
        
        signal, confidence, reasoning = decision_matrix.get(
            (sentiment, stress_level), 
            ("HOLD", 0.5, "Yetersiz veri")
        )
        
        return {
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
            "key_insight": f"CEO {sentiment.lower()} sentiment, {stress_level.lower()} stres"
        }
    
    def generate_emotion_report(self, results: Dict) -> str:
        """Emotion analiz raporu."""
        ticker = results.get("ticker", "UNKNOWN")
        signal = results.get("trading_signal", {})
        stress = results.get("stress_level", {})
        sentiment = results.get("overall_sentiment", {})
        
        signal_emoji = {
            "BUY": "🟢", "STRONG_BUY": "🟢🟢",
            "HOLD": "🟡",
            "SELL": "🔴", "STRONG_SELL": "🔴🔴"
        }
        
        report = f"""
<emotion_analysis>
🎭 CEO DUYGU ANALİZİ - {ticker}
════════════════════════════════════════

{signal_emoji.get(signal.get('signal', 'HOLD'), '⚪')} SİNYAL: {signal.get('signal', 'N/A')}
📊 Güven: %{signal.get('confidence', 0)*100:.0f}
💡 Reasoning: {signal.get('reasoning', 'N/A')}

😰 STRES SEVİYESİ: {stress.get('level', 'N/A')}
  • Göstergeler: {', '.join(stress.get('indicators_found', [])[:3]) or 'Yok'}

💬 SENTIMENT: {sentiment.get('sentiment', 'N/A')}
  • Skor: {sentiment.get('combined_score', 0):.4f}

⚠️ AKSİYON:
  {signal.get('key_insight', 'N/A')}

</emotion_analysis>
"""
        return report
