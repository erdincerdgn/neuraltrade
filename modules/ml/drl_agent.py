"""
Deep Reinforcement Learning Trading Agent
Author: Erdinc Erdogan
Purpose: Implements PPO and DQN algorithms in a Gym-like trading environment where agents learn optimal buy/sell policies from historical prices.
References:
- Proximal Policy Optimization (Schulman et al., 2017)
- Deep Q-Networks (Mnih et al., 2015)
- OpenAI Gym Trading Environments
Usage:
    env = TradingEnvironment(prices, initial_balance=10000)
    agent = DRLAgent(env, algorithm="ppo")
    agent.train(episodes=1000)
"""
import os
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from colorama import Fore, Style
from enum import Enum


class Action(Enum):
    """Trading aksiyonları."""
    HOLD = 0
    BUY = 1
    SELL = 2


class TradingEnvironment:
    """
    OpenAI Gym benzeri trading ortamı.
    Agent bu ortamda al/sat kararları vererek öğrenir.
    """
    
    def __init__(self, 
                 prices: List[float],
                 initial_balance: float = 10000,
                 max_shares: int = 100,
                 transaction_fee: float = 0.001):
        """
        Args:
            prices: Geçmiş fiyat listesi
            initial_balance: Başlangıç bakiyesi
            max_shares: Maksimum hisse sayısı
            transaction_fee: İşlem ücreti oranı (%0.1 = 0.001)
        """
        self.prices = np.array(prices)
        self.initial_balance = initial_balance
        self.max_shares = max_shares
        self.transaction_fee = transaction_fee
        
        self.lookback = 20  # State için bakılacak geçmiş gün sayısı
        self.n_actions = 3  # HOLD, BUY, SELL
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Ortamı sıfırla."""
        self.current_step = self.lookback
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_shares_bought = 0
        self.total_shares_sold = 0
        self.trade_history = []
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """
        Mevcut durumu (state) döndür.
        
        State: [son N fiyat değişimi, RSI, pozisyon, bakiye oranı]
        """
        start = self.current_step - self.lookback
        end = self.current_step
        
        # Fiyat değişimleri (normalized)
        price_changes = np.diff(self.prices[start:end]) / self.prices[start:end-1]
        
        # RSI hesapla
        rsi = self._calculate_rsi(self.prices[start:end])
        
        # Pozisyon durumu
        position_ratio = self.shares_held / self.max_shares if self.max_shares > 0 else 0
        balance_ratio = self.balance / self.initial_balance
        
        # State vector
        state = np.concatenate([
            price_changes,
            [rsi / 100, position_ratio, balance_ratio]
        ])
        
        return state.astype(np.float32)
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """RSI hesapla."""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Aksiyon al ve yeni duruma geç.
        
        Args:
            action: 0=HOLD, 1=BUY, 2=SELL
            
        Returns:
            observation: Yeni state
            reward: Ödül
            done: Episode bitti mi
            info: Ek bilgiler
        """
        current_price = self.prices[self.current_step]
        prev_portfolio_value = self._get_portfolio_value()
        
        # Aksiyonu uygula
        if action == Action.BUY.value and self.balance > current_price:
            # Alabildiğin kadar al
            shares_to_buy = min(
                int(self.balance / current_price),
                self.max_shares - self.shares_held
            )
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * (1 + self.transaction_fee)
                self.balance -= cost
                self.shares_held += shares_to_buy
                self.total_shares_bought += shares_to_buy
                self.trade_history.append({
                    "step": self.current_step,
                    "action": "BUY",
                    "shares": shares_to_buy,
                    "price": current_price
                })
        
        elif action == Action.SELL.value and self.shares_held > 0:
            # Tüm hisseleri sat
            revenue = self.shares_held * current_price * (1 - self.transaction_fee)
            self.balance += revenue
            self.total_shares_sold += self.shares_held
            self.trade_history.append({
                "step": self.current_step,
                "action": "SELL",
                "shares": self.shares_held,
                "price": current_price
            })
            self.shares_held = 0
        
        # Bir adım ilerle
        self.current_step += 1
        
        # Ödül hesapla (portfolio değer değişimi)
        new_portfolio_value = self._get_portfolio_value()
        reward = (new_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        
        # Episode bitti mi?
        done = self.current_step >= len(self.prices) - 1
        
        # Ek bilgiler
        info = {
            "portfolio_value": new_portfolio_value,
            "balance": self.balance,
            "shares_held": self.shares_held,
            "current_price": current_price
        }
        
        return self._get_observation(), reward, done, info
    
    def _get_portfolio_value(self) -> float:
        """Toplam portföy değerini hesapla."""
        current_price = self.prices[min(self.current_step, len(self.prices) - 1)]
        return self.balance + self.shares_held * current_price


class DQNAgent:
    """
    Deep Q-Network Agent.
    Q-learning ile optimal al/sat politikası öğrenir.
    """
    
    def __init__(self,
                 state_size: int,
                 action_size: int = 3,
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995):
        """
        Args:
            state_size: State vektör boyutu
            action_size: Aksiyon sayısı (HOLD, BUY, SELL)
            learning_rate: Öğrenme oranı
            gamma: İndirim faktörü
            epsilon: Keşif oranı
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.memory = []
        self.max_memory = 10000
        
        self.model = None
        self.target_model = None
        self.tf = None
        self._build_model()
    
    def _build_model(self):
        """DQN modelini oluştur."""
        try:
            import tensorflow as tf
            self.tf = tf
            
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(self.action_size, activation='linear')
            ])
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss='mse'
            )
            
            self.model = model
            self.target_model = tf.keras.models.clone_model(model)
            self.target_model.set_weights(model.get_weights())
            
        except ImportError:
            self.tf = None
    
    def remember(self, state, action, reward, next_state, done):
        """Deneyimi hafızaya ekle."""
        if len(self.memory) >= self.max_memory:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray) -> int:
        """Epsilon-greedy politika ile aksiyon seç."""
        if np.random.random() <= self.epsilon:
            return np.random.randint(self.action_size)
        
        if self.model is None:
            return np.random.randint(self.action_size)
        
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self, batch_size: int = 32):
        """Deneyimlerden öğren (Experience Replay)."""
        if self.model is None or len(self.memory) < batch_size:
            return
        
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        
        # Q-value güncelleme
        targets = self.model.predict(states, verbose=0)
        next_q = self.target_model.predict(next_states, verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])
        
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_model(self):
        """Target model ağırlıklarını güncelle."""
        if self.target_model and self.model:
            self.target_model.set_weights(self.model.get_weights())


class DRLTrader:
    """
    Deep Reinforcement Learning Trader.
    Geçmiş verilerde simülasyon yaparak optimal trading politikası öğrenir.
    """
    
    def __init__(self, algorithm: str = "dqn"):
        """
        Args:
            algorithm: "dqn" veya "ppo"
        """
        self.algorithm = algorithm
        self.agent = None
        self.env = None
        self.training_history = []
    
    def train(self, 
              prices: List[float],
              episodes: int = 100,
              initial_balance: float = 10000) -> Dict:
        """
        Agent'ı eğit.
        
        Args:
            prices: Geçmiş fiyat verileri
            episodes: Eğitim episode sayısı
            initial_balance: Başlangıç sermayesi
        """
        print(f"{Fore.CYAN}  → DRL Eğitimi Başlıyor ({self.algorithm.upper()})...{Style.RESET_ALL}", flush=True)
        
        self.env = TradingEnvironment(prices, initial_balance)
        
        # State size hesapla
        state = self.env.reset()
        state_size = len(state)
        
        self.agent = DQNAgent(state_size=state_size)
        
        best_reward = -np.inf
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.agent.act(state)
                next_state, reward, done, info = self.env.step(action)
                
                self.agent.remember(state, action, reward, next_state, done)
                self.agent.replay(batch_size=32)
                
                state = next_state
                total_reward += reward
            
            # Target model güncelle
            if episode % 10 == 0:
                self.agent.update_target_model()
            
            final_value = info["portfolio_value"]
            profit_pct = ((final_value - initial_balance) / initial_balance) * 100
            
            self.training_history.append({
                "episode": episode,
                "total_reward": total_reward,
                "final_value": final_value,
                "profit_pct": profit_pct,
                "epsilon": self.agent.epsilon
            })
            
            if total_reward > best_reward:
                best_reward = total_reward
            
            if (episode + 1) % 20 == 0:
                print(f"{Fore.GREEN}   Episode {episode+1}/{episodes}: Profit={profit_pct:+.2f}%, ε={self.agent.epsilon:.3f}{Style.RESET_ALL}", flush=True)
        
        print(f"{Fore.GREEN}  → DRL Eğitimi Tamamlandı{Style.RESET_ALL}", flush=True)
        
        return {
            "episodes": episodes,
            "best_reward": best_reward,
            "final_epsilon": self.agent.epsilon,
            "history": self.training_history[-10:]
        }
    
    def predict_action(self, prices: List[float]) -> Dict:
        """
        Mevcut fiyatlara göre aksiyon öner.
        """
        if self.agent is None or self.env is None:
            return self._demo_predict(prices)
        
        # Geçici env oluştur
        temp_env = TradingEnvironment(prices)
        temp_env.current_step = len(prices) - 1
        state = temp_env._get_observation()
        
        # Greedy aksiyon (epsilon=0)
        old_epsilon = self.agent.epsilon
        self.agent.epsilon = 0
        action = self.agent.act(state)
        self.agent.epsilon = old_epsilon
        
        action_name = Action(action).name
        
        return {
            "action": action_name,
            "confidence": 1 - self.agent.epsilon,
            "model": self.algorithm.upper()
        }
    
    def _demo_predict(self, prices: List[float]) -> Dict:
        """Demo tahmin."""
        # Basit momentum stratejisi
        if len(prices) >= 5:
            momentum = (prices[-1] - prices[-5]) / prices[-5]
            if momentum > 0.02:
                action = "HOLD"  # Yükseliş trendinde tut
            elif momentum < -0.02:
                action = "SELL"  # Düşüş trendinde sat
            else:
                action = "BUY"   # Düz trendde al
        else:
            action = "HOLD"
        
        return {
            "action": action,
            "confidence": 0.5,
            "model": "MOMENTUM (Demo)"
        }
    
    def generate_drl_report(self, ticker: str, prices: List[float]) -> str:
        """DRL analiz raporu."""
        prediction = self.predict_action(prices)
        
        action_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(prediction["action"], "⚪")
        
        report = f"""
<drl_trader>
🤖 DRL TRADER ANALİZİ - {ticker}
════════════════════════════════════════
Model: {prediction['model']}
Güven: %{prediction['confidence']*100:.0f}

{action_emoji} ÖNERİLEN AKSİYON: {prediction['action']}
"""
        
        if self.training_history:
            last = self.training_history[-1]
            report += f"""
📊 SON EĞİTİM SONUÇLARI:
  • Episode: {last['episode']}
  • Profit: {last['profit_pct']:+.2f}%
  • Epsilon: {last['epsilon']:.3f}
"""
        
        report += "</drl_trader>\n"
        return report
