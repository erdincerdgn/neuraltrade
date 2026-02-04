# NeuralTrade Mobile

**Layer 1: Presentation & Client Layer** - React Native/Expo Mobile App

AI-powered trading platform mobile application built with Expo, featuring real-time market data, AI trading signals, and portfolio management.

## 🏗️ Architecture

```
Layer 1: Mobile App (Expo/React Native)
    ↓ HTTPS / WebSocket
Layer 2: API Gateway (Nginx)
    ↓
Layer 3: NestJS Backend (Port 4000)
    ↓ gRPC
Layer 4: Python AI Engine (Port 8000/50051)
    ↓
Layer 5: Data Persistence (PostgreSQL, Redis)
    ↓
Layer 6: Vector DB & Observability (Qdrant, Prometheus)
```

## 🚀 Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm start

# Run on specific platform
npm run ios      # iOS Simulator
npm run android  # Android Emulator
npm run web      # Web Browser
```

## 📁 Project Structure

```
neuraltrade-mobile/
├── app/                    # Expo Router file-based routing
│   ├── _layout.tsx         # Root layout with providers
│   ├── modal.tsx           # Modal screen
│   └── (tabs)/             # Bottom tab navigation
│       ├── _layout.tsx     # Tab bar configuration
│       ├── index.tsx       # Dashboard screen
│       ├── ai-signals.tsx  # AI trading signals
│       ├── trade.tsx       # Order execution
│       └── portfolio.tsx   # Positions & P&L
├── components/             # Reusable UI components
│   ├── screen-wrapper.tsx  # SafeArea + Dark theme wrapper
│   ├── signal-card.tsx     # AI signal display card
│   └── ui/                 # Base UI components
├── hooks/                  # Custom React hooks
│   ├── use-socket.ts       # WebSocket connection hook
│   └── use-market-data.ts  # React Query market data hook
├── store/                  # Zustand state management
│   ├── market.store.ts     # Real-time market data
│   ├── portfolio.store.ts  # Portfolio state (persisted)
│   └── signals.store.ts    # AI signals state
├── services/               # API & Socket services
│   ├── api.ts              # Axios client for REST API
│   └── socket.ts           # Socket.io for real-time
├── constants/              # Theme & configuration
│   └── theme.ts            # Neural-Dark theme colors
└── config files
    ├── tailwind.config.js  # NativeWind configuration
    ├── metro.config.js     # Metro bundler with NativeWind
    └── babel.config.js     # Babel with Reanimated plugin
```

## 🎨 Design System

### Neural-Dark Theme

| Element         | Color     | Usage                |
|-----------------|-----------|----------------------|
| Background      | `#000000` | Screen backgrounds   |
| Card            | `#1A1A1A` | Card surfaces        |
| Border          | `#2A2A2A` | Subtle borders       |
| Primary (Matrix)| `#00FF41` | Accents, bullish     |
| Bearish         | `#FF3B30` | Sell signals, losses |
| Neutral         | `#FFD60A` | Warnings, hold       |

### Typography

- **Headings**: System font, bold weight
- **Terminal**: Monospace for agent thoughts
- **Body**: System default

## 🔌 Backend Integration

### REST API (Layer 3 - NestJS)

```typescript
// services/api.ts
const API_URL = 'http://localhost:4000/api/v1';

// Endpoints
/auth/login          # Authentication
/portfolio/summary   # Portfolio data
/market/quote/:sym   # Market quotes
/trade/order         # Order execution
/signals/latest      # AI signals
```

### WebSocket (Real-time)

```typescript
// hooks/use-socket.ts
const socket = io('http://localhost:4000');

// Events
socket.on('market:tick', (tick) => {});
socket.on('signal:new', (signal) => {});
socket.on('agent:thought', (thought) => {});
```

## 📦 Key Dependencies

| Package                | Purpose                          |
|------------------------|----------------------------------|
| `expo-router`          | File-based navigation            |
| `nativewind`           | Tailwind CSS for React Native    |
| `lucide-react-native`  | Icons                            |
| `socket.io-client`     | Real-time WebSocket              |
| `@tanstack/react-query`| Server state management          |
| `zustand`              | Client state management          |
| `react-native-reanimated` | Smooth animations             |
| `react-native-webview` | TradingView chart embedding      |

## 🛡️ SafeArea Handling

All screens use `ScreenWrapper` component for consistent safe area handling:

```tsx
import { ScreenWrapper } from '@/components/screen-wrapper';

export default function MyScreen() {
  return (
    <ScreenWrapper>
      {/* Content automatically respects notch/home indicator */}
    </ScreenWrapper>
  );
}
```

## 📡 Real-time Optimization

The `useSocket` hook is optimized for high-frequency updates:

```tsx
// Memoized listeners prevent unnecessary re-renders
const { subscribeToSymbol, isConnected } = useSocket({
  autoConnect: true,
  enableSignals: true,
  enableAgentThoughts: showTerminal,
});
```

## 🧪 Development

### Environment Variables

Create `.env` in project root:

```bash
EXPO_PUBLIC_API_URL=http://localhost:4000/api/v1
EXPO_PUBLIC_WS_URL=http://localhost:4000
EXPO_PUBLIC_AI_URL=http://localhost:8000
```

### Clear Cache

```bash
npx expo start --clear
```

### Build for Production

```bash
# Build for iOS
eas build --platform ios

# Build for Android
eas build --platform android
```

## 📱 Screens

### Dashboard
- Portfolio value summary
- Day P&L with trend indicator
- Quick stats (signals, positions, orders)
- Real-time watchlist

### AI Signals
- Live AI-generated trading signals
- Confidence scores and reasoning
- Agent terminal view (swarm thoughts)
- Filter by action type

### Trade
- Symbol selection with mini chart
- Buy/Sell toggle
- Order types (Market, Limit, Stop)
- Stop Loss & Take Profit
- Order cost summary

### Portfolio
- Total value and P&L tracking
- Open positions list
- Pending orders
- Trade history

## 🔗 Related Services

- **NestJS Backend**: `neuraltrade-be/` (Port 4000)
- **Python AI Engine**: `main.py` (Port 8000, gRPC 50051)
- **Admin Panel**: `neuraltrade-admin/` (Port 3010)
- **Web Frontend**: `neuraltrade-fe/` (Port 3001)

## 📄 License

Private - NeuralTrade
