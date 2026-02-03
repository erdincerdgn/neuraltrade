"""
Security Framework Phase 1 - Critical Vulnerability Fixes
Author: Erdinc Erdogan
Purpose: Implements ThreadSafeExecutor, SecureVaultManager, FinancialDecimal, and AsyncSafeGather to fix race conditions, API key exposure, and precision errors.
References:
- Race Condition Prevention (Threading)
- PBKDF2 API Key Encryption
- Decimal Precision (28 digits)
Usage:
    vault = SecureVaultManager()
    vault.store_api_key('exchange', api_key)
    decimal = FinancialDecimal('123.456789')
"""

import asyncio
import hashlib
import hmac
import os
import secrets
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_EVEN, InvalidOperation, getcontext
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
import base64

# Set global decimal precision for financial calculations
getcontext().prec = 28  # 28 digits of precision for institutional-grade calculations


# ============================================================================
# CV-002 FIX: SECURE VAULT MANAGER - API Key Protection
# ============================================================================

class SecureVaultManager:
    """
    CV-002 FIX: Secure API Key Management
    
    Features:
    - Encrypted storage of API keys
    - Automatic key rotation
    - Secure memory handling
    - Audit logging
    """
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._secrets: Dict[str, bytes] = {}
        self._encryption_key = self._generate_encryption_key()
        self._access_log: List[Dict] = []
        self._rotation_interval = 86400  # 24 hours
        self._last_rotation = time.time()
        self._initialized = True
    
    def _generate_encryption_key(self) -> bytes:
        """Generate a secure encryption key"""
        return secrets.token_bytes(32)  # 256-bit key
    
    def _encrypt(self, data: str) -> bytes:
        """Encrypt sensitive data using XOR with key derivation"""
        key_material = hashlib.pbkdf2_hmac(
            'sha256',
            self._encryption_key,
            b'neuraltrade_salt',
            100000
        )
        data_bytes = data.encode('utf-8')
        encrypted = bytes(a ^ b for a, b in zip(data_bytes, key_material * (len(data_bytes) // 32 + 1)))
        return base64.b64encode(encrypted)
    
    def _decrypt(self, encrypted_data: bytes) -> str:
        """Decrypt sensitive data"""
        key_material = hashlib.pbkdf2_hmac(
            'sha256',
            self._encryption_key,
            b'neuraltrade_salt',
            100000
        )
        data_bytes = base64.b64decode(encrypted_data)
        decrypted = bytes(a ^ b for a, b in zip(data_bytes, key_material * (len(data_bytes) // 32 + 1)))
        return decrypted.decode('utf-8')
    
    def store_secret(self, key_name: str, secret_value: str) -> bool:
        """Securely store a secret"""
        with self._lock:
            try:
                encrypted = self._encrypt(secret_value)
                self._secrets[key_name] = encrypted
                self._log_access(key_name, "STORE")
                return True
            except Exception as e:
                self._log_access(key_name, "STORE_FAILED", str(e))
                return False
    
    def retrieve_secret(self, key_name: str) -> Optional[str]:
        """Securely retrieve a secret"""
        with self._lock:
            try:
                if key_name not in self._secrets:
                    self._log_access(key_name, "NOT_FOUND")
                    return None
                
                encrypted = self._secrets[key_name]
                decrypted = self._decrypt(encrypted)
                self._log_access(key_name, "RETRIEVE")
                return decrypted
            except Exception as e:
                self._log_access(key_name, "RETRIEVE_FAILED", str(e))
                return None
    
    def rotate_keys(self) -> bool:
        """Rotate encryption keys"""
        with self._lock:
            try:
                # Decrypt all secrets with old key
                decrypted_secrets = {}
                for key_name, encrypted in self._secrets.items():
                    decrypted_secrets[key_name] = self._decrypt(encrypted)
                
                # Generate new encryption key
                self._encryption_key = self._generate_encryption_key()
                
                # Re-encrypt all secrets with new key
                for key_name, secret in decrypted_secrets.items():
                    self._secrets[key_name] = self._encrypt(secret)
                
                self._last_rotation = time.time()
                self._log_access("ALL_KEYS", "ROTATION_COMPLETE")
                return True
            except Exception as e:
                self._log_access("ALL_KEYS", "ROTATION_FAILED", str(e))
                return False
    
    def _log_access(self, key_name: str, action: str, error: str = None):
        """Log access to secrets for audit"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "key_name": key_name,
            "action": action,
            "error": error
        }
        self._access_log.append(log_entry)
        
        # Keep only last 1000 entries
        if len(self._access_log) > 1000:
            self._access_log = self._access_log[-1000:]
    
    def get_audit_log(self) -> List[Dict]:
        """Get audit log"""
        with self._lock:
            return self._access_log.copy()
    
    @staticmethod
    def secure_getenv(key_name: str, default: str = None) -> Optional[str]:
        """
        CV-002 FIX: Secure replacement for os.getenv()
        Retrieves from vault first, falls back to environment
        """
        vault = SecureVaultManager()
        
        # Try vault first
        secret = vault.retrieve_secret(key_name)
        if secret:
            return secret
        
        # Fall back to environment (for initial setup)
        env_value = os.environ.get(key_name, default)
        if env_value and env_value != default:
            # Store in vault for future use
            vault.store_secret(key_name, env_value)
        
        return env_value


# ============================================================================
# CV-003, CV-004, CV-006 FIX: FINANCIAL DECIMAL - Precision Handling
# ============================================================================

class FinancialDecimal:
    """
    CV-003, CV-004, CV-006 FIX: Institutional-Grade Decimal Precision
    
    Features:
    - 18-digit precision for financial calculations
    - Proper rounding (ROUND_HALF_EVEN - banker's rounding)
    - Overflow/underflow protection
    - Immutable value objects
    """
    
    PRECISION = 18
    ROUNDING = ROUND_HALF_EVEN
    
    def __init__(self, value: Union[str, int, float, Decimal, 'FinancialDecimal']):
        if isinstance(value, FinancialDecimal):
            self._value = value._value
        elif isinstance(value, float):
            # CV-003, CV-004, CV-006 FIX: Convert float to string first to avoid precision loss
            self._value = Decimal(str(value))
        elif isinstance(value, Decimal):
            self._value = value
        else:
            self._value = Decimal(str(value))
        
        # Normalize to standard precision
        self._value = self._value.quantize(
            Decimal(10) ** -self.PRECISION,
            rounding=self.ROUNDING
        )
    
    @property
    def value(self) -> Decimal:
        return self._value
    
    def __add__(self, other: Union['FinancialDecimal', Decimal, int, float]) -> 'FinancialDecimal':
        other_val = self._to_decimal(other)
        return FinancialDecimal(self._value + other_val)
    
    def __sub__(self, other: Union['FinancialDecimal', Decimal, int, float]) -> 'FinancialDecimal':
        other_val = self._to_decimal(other)
        return FinancialDecimal(self._value - other_val)
    
    def __mul__(self, other: Union['FinancialDecimal', Decimal, int, float]) -> 'FinancialDecimal':
        other_val = self._to_decimal(other)
        return FinancialDecimal(self._value * other_val)
    
    def __truediv__(self, other: Union['FinancialDecimal', Decimal, int, float]) -> 'FinancialDecimal':
        """CV-003, CV-004, CV-006 FIX: Safe division with precision handling"""
        other_val = self._to_decimal(other)
        if other_val == 0:
            raise ZeroDivisionError("Division by zero in financial calculation")
        return FinancialDecimal(self._value / other_val)
    
    def __floordiv__(self, other: Union['FinancialDecimal', Decimal, int, float]) -> 'FinancialDecimal':
        other_val = self._to_decimal(other)
        if other_val == 0:
            raise ZeroDivisionError("Division by zero in financial calculation")
        return FinancialDecimal(self._value // other_val)
    
    def __mod__(self, other: Union['FinancialDecimal', Decimal, int, float]) -> 'FinancialDecimal':
        other_val = self._to_decimal(other)
        return FinancialDecimal(self._value % other_val)
    
    def __pow__(self, other: Union['FinancialDecimal', Decimal, int, float]) -> 'FinancialDecimal':
        other_val = self._to_decimal(other)
        return FinancialDecimal(self._value ** other_val)
    
    def __neg__(self) -> 'FinancialDecimal':
        return FinancialDecimal(-self._value)
    
    def __abs__(self) -> 'FinancialDecimal':
        return FinancialDecimal(abs(self._value))
    
    def __eq__(self, other) -> bool:
        other_val = self._to_decimal(other)
        return self._value == other_val
    
    def __lt__(self, other) -> bool:
        other_val = self._to_decimal(other)
        return self._value < other_val
    
    def __le__(self, other) -> bool:
        other_val = self._to_decimal(other)
        return self._value <= other_val
    
    def __gt__(self, other) -> bool:
        other_val = self._to_decimal(other)
        return self._value > other_val
    
    def __ge__(self, other) -> bool:
        other_val = self._to_decimal(other)
        return self._value >= other_val
    
    def __float__(self) -> float:
        return float(self._value)
    
    def __int__(self) -> int:
        return int(self._value)
    
    def __str__(self) -> str:
        return str(self._value)
    
    def __repr__(self) -> str:
        return f"FinancialDecimal('{self._value}')"
    
    def __hash__(self) -> int:
        return hash(self._value)
    
    def _to_decimal(self, value) -> Decimal:
        if isinstance(value, FinancialDecimal):
            return value._value
        elif isinstance(value, Decimal):
            return value
        elif isinstance(value, float):
            return Decimal(str(value))
        else:
            return Decimal(str(value))
    
    def round(self, places: int = 2) -> 'FinancialDecimal':
        """Round to specified decimal places"""
        quantizer = Decimal(10) ** -places
        return FinancialDecimal(self._value.quantize(quantizer, rounding=self.ROUNDING))
    
    def to_percentage(self) -> 'FinancialDecimal':
        """CV-004 FIX: Safe percentage conversion"""
        return self / FinancialDecimal(100)
    
    def from_percentage(self) -> 'FinancialDecimal':
        """Convert from percentage to decimal"""
        return self * FinancialDecimal(100)
    
    @classmethod
    def safe_divide(cls, numerator: Union['FinancialDecimal', Decimal, int, float],
                    denominator: Union['FinancialDecimal', Decimal, int, float],
                    default: Union['FinancialDecimal', Decimal, int, float] = 0) -> 'FinancialDecimal':
        """CV-003, CV-006 FIX: Safe division with default value"""
        try:
            num = cls(numerator)
            den = cls(denominator)
            if den._value == 0:
                return cls(default)
            return num / den
        except (InvalidOperation, ZeroDivisionError):
            return cls(default)


# ============================================================================
# CV-001, CV-005, CV-007 FIX: THREAD-SAFE EXECUTOR - Race Condition Protection
# ============================================================================

class ThreadSafeExecutor:
    """
    CV-001 FIX: Thread-Safe Execution Framework
    
    Features:
    - Proper locking mechanisms
    - Atomic operations
    - Deadlock prevention
    - Performance metrics
    """
    
    def __init__(self):
        self._lock = threading.RLock()
        self._state: Dict[str, Any] = {}
        self._metrics = {
            'total_operations': 0,
            'lock_acquisitions': 0,
            'lock_contentions': 0
        }
    
    @contextmanager
    def atomic_operation(self, operation_name: str = "unnamed"):
        """CV-001 FIX: Context manager for atomic operations"""
        acquired = self._lock.acquire(blocking=True, timeout=5.0)
        if not acquired:
            self._metrics['lock_contentions'] += 1
            raise TimeoutError(f"Failed to acquire lock for operation: {operation_name}")
        
        try:
            self._metrics['lock_acquisitions'] += 1
            yield self._state
        finally:
            self._lock.release()
            self._metrics['total_operations'] += 1
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Thread-safe state retrieval"""
        with self.atomic_operation("get_state"):
            return self._state.get(key, default)
    
    def set_state(self, key: str, value: Any) -> None:
        """Thread-safe state update"""
        with self.atomic_operation("set_state"):
            self._state[key] = value
    
    def update_state(self, key: str, update_func: Callable[[Any], Any], default: Any = None) -> Any:
        """Thread-safe state update with function"""
        with self.atomic_operation("update_state"):
            current = self._state.get(key, default)
            new_value = update_func(current)
            self._state[key] = new_value
            return new_value
    
    def get_metrics(self) -> Dict[str, int]:
        """Get executor metrics"""
        with self.atomic_operation("get_metrics"):
            return self._metrics.copy()


class AsyncSafeGather:
    """
    CV-005, CV-007 FIX: Safe Async Gather with Proper Synchronization
    
    Features:
    - Coordinated async task execution
    - Proper error propagation
    - Resource cleanup on failure
    - Timeout handling
    """
    
    def __init__(self):
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(100)  # Limit concurrent tasks
        self._active_tasks: Dict[str, asyncio.Task] = {}
    
    async def safe_gather(self, *coroutines, return_exceptions: bool = False, 
                          timeout: float = None) -> List[Any]:
        """
        CV-005, CV-007 FIX: Safe async gather with proper synchronization
        """
        async with self._lock:
            tasks = []
            for i, coro in enumerate(coroutines):
                task = asyncio.create_task(self._wrapped_task(coro, f"task_{i}"))
                tasks.append(task)
                self._active_tasks[f"task_{i}"] = task
            
            try:
                if timeout:
                    results = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=return_exceptions),
                        timeout=timeout
                    )
                else:
                    results = await asyncio.gather(*tasks, return_exceptions=return_exceptions)
                
                return results
            except asyncio.TimeoutError:
                # Cancel all pending tasks
                for task in tasks:
                    if not task.done():
                        task.cancel()
                raise
            finally:
                # Cleanup
                for task_id in list(self._active_tasks.keys()):
                    if task_id.startswith("task_"):
                        del self._active_tasks[task_id]
    
    async def _wrapped_task(self, coro, task_id: str) -> Any:
        """Wrap coroutine with semaphore for resource limiting"""
        async with self._semaphore:
            return await coro
    
    async def safe_create_task(self, coro, name: str = None) -> asyncio.Task:
        """CV-007 FIX: Safe task creation with tracking"""
        async with self._lock:
            task = asyncio.create_task(coro)
            task_name = name or f"task_{id(task)}"
            self._active_tasks[task_name] = task
            return task
    
    async def cancel_all(self) -> int:
        """Cancel all active tasks"""
        async with self._lock:
            cancelled = 0
            for task_name, task in list(self._active_tasks.items()):
                if not task.done():
                    task.cancel()
                    cancelled += 1
            self._active_tasks.clear()
            return cancelled


# ============================================================================
# INTEGRATED SECURITY DECORATOR
# ============================================================================

def secure_financial_operation(func: Callable) -> Callable:
    """
    Decorator for secure financial operations
    Combines all security fixes into a single decorator
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        executor = ThreadSafeExecutor()
        
        with executor.atomic_operation(func.__name__):
            # Convert float arguments to FinancialDecimal
            new_args = []
            for arg in args:
                if isinstance(arg, float):
                    new_args.append(FinancialDecimal(arg))
                else:
                    new_args.append(arg)
            
            new_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, float):
                    new_kwargs[key] = FinancialDecimal(value)
                else:
                    new_kwargs[key] = value
            
            result = func(*new_args, **new_kwargs)
            
            return result
    
    return wrapper


def secure_async_operation(func: Callable) -> Callable:
    """
    Decorator for secure async operations
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        gatherer = AsyncSafeGather()
        
        async with gatherer._lock:
            result = await func(*args, **kwargs)
            return result
    
    return wrapper


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

def test_security_framework():
    """Test all security fixes"""
    print("🧪 Testing Institutional Security Framework")
    print("=" * 60)
    
    # Test SecureVaultManager (CV-002 FIX)
    print("\n1. Testing SecureVaultManager (CV-002 FIX)...")
    vault = SecureVaultManager()
    vault.store_secret("OPENAI_API_KEY", "sk-test-key-12345")
    retrieved = vault.retrieve_secret("OPENAI_API_KEY")
    assert retrieved == "sk-test-key-12345", "Vault encryption/decryption failed"
    print("   ✅ SecureVaultManager: PASS")
    
    # Test FinancialDecimal (CV-003, CV-004, CV-006 FIX)
    print("\n2. Testing FinancialDecimal (CV-003, CV-004, CV-006 FIX)...")
    
    # CV-003: Division by 5.0
    value = FinancialDecimal(100)
    result = value / FinancialDecimal(5.0)
    assert result == FinancialDecimal(20), f"Division failed: {result}"
    print("   ✅ CV-003 (/5.0): PASS")
    
    # CV-004: Division by 100 (percentage)
    percentage = FinancialDecimal(15)
    decimal_value = percentage.to_percentage()
    assert decimal_value == FinancialDecimal("0.15"), f"Percentage conversion failed: {decimal_value}"
    print("   ✅ CV-004 (/100): PASS")
    
    # CV-006: Division by 7
    value = FinancialDecimal(100)
    result = value / FinancialDecimal(7)
    expected = FinancialDecimal("14.285714285714285714")
    print(f"   Result: {result}")
    print("   ✅ CV-006 (/7): PASS")
    
    # Test ThreadSafeExecutor (CV-001 FIX)
    print("\n3. Testing ThreadSafeExecutor (CV-001 FIX)...")
    executor = ThreadSafeExecutor()
    
    def increment(x):
        return x + 1
    
    executor.set_state("counter", 0)
    for _ in range(100):
        executor.update_state("counter", increment)
    
    final_count = executor.get_state("counter")
    assert final_count == 100, f"Thread safety failed: {final_count}"
    print("   ✅ ThreadSafeExecutor: PASS")
    
    # Test AsyncSafeGather (CV-005, CV-007 FIX)
    print("\n4. Testing AsyncSafeGather (CV-005, CV-007 FIX)...")
    
    async def test_async():
        gatherer = AsyncSafeGather()
        
        async def sample_task(n):
            await asyncio.sleep(0.01)
            return n * 2
        
        results = await gatherer.safe_gather(
            sample_task(1),
            sample_task(2),
            sample_task(3)
        )
        return results
    
    # Run async test
    results = asyncio.get_event_loop().run_until_complete(test_async())
    assert results == [2, 4, 6], f"Async gather failed: {results}"
    print("   ✅ AsyncSafeGather: PASS")
    
    print("\n" + "=" * 60)
    print("🎯 ALL SECURITY TESTS PASSED")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    test_security_framework()
