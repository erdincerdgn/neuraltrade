"""
Redis Signal Publisher
Author: Erdinc Erdogan
Purpose: Publishes AI trading signals to Redis pub/sub for real-time consumption by NestJS backend with graceful degradation when Redis is unavailable.
References:
- Redis Pub/Sub Pattern
- Real-time Signal Streaming
- Python-NestJS Communication
Usage:
    publisher = SignalPublisher()
    publisher.publish_signal({"symbol": "BTC/USDT", "action": "BUY", "confidence": 0.85})
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

# ============================================
# GRACEFUL REDIS IMPORT
# ============================================

REDIS_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
    logger.info("✅ Redis loaded successfully")
except ImportError:
    redis = None
    logger.warning("⚠️ Redis not installed. Signal publishing disabled.")


class SignalPublisher:
    """
    Redis-based signal publisher for Python → NestJS communication.
    
    Publishes to 'ai:signals' channel for real-time signal streaming.
    Gracefully degrades when Redis is not available.
    """
    
    # Redis channel for AI signals
    SIGNAL_CHANNEL = 'ai:signals'
    
    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize Redis connection.
        
        Args:
            redis_url: Redis connection URL. Defaults to REDIS_URL env var.
        """
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379')
        self._client = None
        
    @property
    def client(self):
        """Lazy-load Redis client."""
        if not REDIS_AVAILABLE:
            return None
        if self._client is None:
            self._client = redis.from_url(self.redis_url, decode_responses=True)
        return self._client
    
    def publish_signal(
        self,
        symbol: str,
        action: str,
        confidence: float,
        models: Optional[List[str]] = None,
        reasoning: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Publish an AI trading signal to Redis.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            action: Signal action - 'BUY', 'SELL', or 'HOLD'
            confidence: Confidence score 0.0 to 1.0
            models: List of models that contributed to signal
            reasoning: Human-readable explanation
            metadata: Additional signal metadata
            
        Returns:
            True if published successfully, False otherwise.
        """
        # Check if Redis is available
        if not REDIS_AVAILABLE:
            logger.debug(f"[Mock] Signal: {symbol} {action} ({confidence:.1%})")
            return False
        
        try:
            signal = {
                'symbol': symbol,
                'action': action.upper(),
                'confidence': confidence,
                'models': models or ['pipeline'],
                'reasoning': reasoning or f'{action} signal for {symbol}',
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'metadata': metadata or {},
            }
            
            # Publish to Redis channel
            if self.client:
                self.client.publish(self.SIGNAL_CHANNEL, json.dumps(signal))
                logger.info(f"📡 Signal published: {symbol} {action} ({confidence:.1%})")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to publish signal: {e}")
            return False
    
    def publish_batch_signals(self, signals: List[Dict[str, Any]]) -> int:
        """
        Publish multiple signals.
        
        Args:
            signals: List of signal dictionaries.
            
        Returns:
            Number of signals successfully published.
        """
        success_count = 0
        for signal in signals:
            if self.publish_signal(**signal):
                success_count += 1
        return success_count
    
    def health_check(self) -> bool:
        """Check Redis connection health."""
        if not REDIS_AVAILABLE:
            return False
        try:
            return self.client.ping() if self.client else False
        except Exception:
            return False
    
    def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            self._client.close()
            self._client = None


# ============================================
# SINGLETON INSTANCE
# ============================================

_publisher_instance: Optional[SignalPublisher] = None


def get_signal_publisher() -> SignalPublisher:
    """Get singleton SignalPublisher instance."""
    global _publisher_instance
    if _publisher_instance is None:
        _publisher_instance = SignalPublisher()
    return _publisher_instance


# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

def publish_buy_signal(
    symbol: str,
    confidence: float,
    reasoning: str = "",
    models: Optional[List[str]] = None,
) -> bool:
    """Convenience function to publish a BUY signal."""
    return get_signal_publisher().publish_signal(
        symbol=symbol,
        action='BUY',
        confidence=confidence,
        models=models,
        reasoning=reasoning,
    )


def publish_sell_signal(
    symbol: str,
    confidence: float,
    reasoning: str = "",
    models: Optional[List[str]] = None,
) -> bool:
    """Convenience function to publish a SELL signal."""
    return get_signal_publisher().publish_signal(
        symbol=symbol,
        action='SELL',
        confidence=confidence,
        models=models,
        reasoning=reasoning,
    )


def publish_hold_signal(
    symbol: str,
    confidence: float,
    reasoning: str = "",
) -> bool:
    """Convenience function to publish a HOLD signal."""
    return get_signal_publisher().publish_signal(
        symbol=symbol,
        action='HOLD',
        confidence=confidence,
        reasoning=reasoning,
    )


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    publisher = SignalPublisher()
    
    if publisher.health_check():
        print("✅ Redis connected")
        
        # Example signal
        publisher.publish_signal(
            symbol='BTC/USDT',
            action='BUY',
            confidence=0.87,
            models=['neural_forecaster', 'regime_detector', 'swarm_consensus'],
            reasoning='Strong bullish divergence with increasing volume. RSI recovering from oversold.',
        )
    else:
        print("❌ Redis not available")
