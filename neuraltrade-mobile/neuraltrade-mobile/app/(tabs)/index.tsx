/**
 * Dashboard Screen
 * =================
 * Main screen showing market overview and watchlist
 * Real-time updates via WebSocket from Layer 3
 */

import React, { useEffect } from 'react';
import { View, Text, StyleSheet, ScrollView, Pressable } from 'react-native';
import Animated, { FadeIn } from 'react-native-reanimated';
import { Wifi, WifiOff, TrendingUp, TrendingDown, Activity } from 'lucide-react-native';

import { ScreenWrapper } from '@/components/common/screen-wrapper/screen-wrapper';
import { useSocket } from '@/hooks/use-socket';
import { useMarketStore, selectWatchlist, selectIsConnected } from '@/store';
import { MATRIX_GREEN, STATUS_BEARISH, NEURAL_GRAY, NEURAL_BORDER } from '@/constants/theme';

// ============================================
// DASHBOARD SCREEN
// ============================================

export default function DashboardScreen() {
  const { subscribeToSymbol } = useSocket({ autoConnect: true });
  const isConnected = useMarketStore(selectIsConnected);
  const watchlist = useMarketStore(selectWatchlist);

  // Subscribe to default symbols on mount
  useEffect(() => {
    const defaultSymbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'BTC-USD'];
    defaultSymbols.forEach((symbol) => subscribeToSymbol(symbol));
  }, [subscribeToSymbol]);

  return (
    <ScreenWrapper>
      <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
        {/* Header */}
        <View style={styles.header}>
          <View>
            <Text style={styles.greeting}>Welcome back</Text>
            <Text style={styles.title}>NeuralTrade</Text>
          </View>
          <View style={styles.connectionStatus}>
            {isConnected ? (
              <Wifi size={20} color={MATRIX_GREEN} />
            ) : (
              <WifiOff size={20} color="#6B7280" />
            )}
            <Text style={[styles.statusText, { color: isConnected ? MATRIX_GREEN : '#6B7280' }]}>
              {isConnected ? 'Live' : 'Offline'}
            </Text>
          </View>
        </View>

        {/* Portfolio Summary Card */}
        <Animated.View entering={FadeIn.delay(100)} style={styles.summaryCard}>
          <Text style={styles.summaryLabel}>Total Portfolio Value</Text>
          <Text style={styles.summaryValue}>$124,567.89</Text>
          <View style={styles.changeRow}>
            <TrendingUp size={16} color={MATRIX_GREEN} />
            <Text style={[styles.changeText, { color: MATRIX_GREEN }]}>
              +$2,345.67 (+1.92%)
            </Text>
            <Text style={styles.changeLabel}>Today</Text>
          </View>
        </Animated.View>

        {/* Quick Stats */}
        <Animated.View entering={FadeIn.delay(200)} style={styles.statsRow}>
          <View style={styles.statCard}>
            <Activity size={20} color={MATRIX_GREEN} />
            <Text style={styles.statValue}>12</Text>
            <Text style={styles.statLabel}>Active Signals</Text>
          </View>
          <View style={styles.statCard}>
            <TrendingUp size={20} color={MATRIX_GREEN} />
            <Text style={styles.statValue}>8</Text>
            <Text style={styles.statLabel}>Open Positions</Text>
          </View>
          <View style={styles.statCard}>
            <TrendingDown size={20} color={STATUS_BEARISH} />
            <Text style={styles.statValue}>3</Text>
            <Text style={styles.statLabel}>Pending Orders</Text>
          </View>
        </Animated.View>

        {/* Watchlist */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Watchlist</Text>
          {watchlist.length === 0 ? (
            <>
              {/* Demo watchlist items */}
              <DemoWatchlistItem
                symbol="AAPL"
                name="Apple Inc."
                price={178.45}
                change={2.34}
                changePercent={1.33}
              />
              <DemoWatchlistItem
                symbol="TSLA"
                name="Tesla Inc."
                price={242.68}
                change={-5.21}
                changePercent={-2.10}
              />
              <DemoWatchlistItem
                symbol="BTC-USD"
                name="Bitcoin"
                price={42567.89}
                change={1234.56}
                changePercent={2.99}
              />
            </>
          ) : (
            watchlist.map((item, index) => (
              <Animated.View
                key={item.symbol}
                entering={FadeIn.delay(300 + index * 50)}
              >
                <WatchlistItem item={item} />
              </Animated.View>
            ))
          )}
        </View>
      </ScrollView>
    </ScreenWrapper>
  );
}

// ============================================
// SUB-COMPONENTS
// ============================================

interface WatchlistItemProps {
  item: {
    symbol: string;
    name: string;
    price: number;
    change: number;
    changePercent: number;
  };
}

function WatchlistItem({ item }: WatchlistItemProps) {
  const isPositive = item.change >= 0;
  const changeColor = isPositive ? MATRIX_GREEN : STATUS_BEARISH;

  return (
    <Pressable style={styles.watchlistItem}>
      <View style={styles.watchlistLeft}>
        <Text style={styles.watchlistSymbol}>{item.symbol}</Text>
        <Text style={styles.watchlistName}>{item.name}</Text>
      </View>
      <View style={styles.watchlistRight}>
        <Text style={styles.watchlistPrice}>${item.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</Text>
        <View style={[styles.changeBadge, { backgroundColor: changeColor + '20' }]}>
          {isPositive ? (
            <TrendingUp size={12} color={changeColor} />
          ) : (
            <TrendingDown size={12} color={changeColor} />
          )}
          <Text style={[styles.changePercent, { color: changeColor }]}>
            {isPositive ? '+' : ''}{item.changePercent.toFixed(2)}%
          </Text>
        </View>
      </View>
    </Pressable>
  );
}

function DemoWatchlistItem({
  symbol,
  name,
  price,
  change,
  changePercent,
}: {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
}) {
  return (
    <WatchlistItem
      item={{ symbol, name, price, change, changePercent }}
    />
  );
}

// ============================================
// STYLES
// ============================================

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 24,
    marginTop: 8,
  },
  greeting: {
    fontSize: 14,
    color: '#6B7280',
  },
  title: {
    fontSize: 28,
    fontWeight: '700',
    color: '#FFFFFF',
  },
  connectionStatus: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: NEURAL_GRAY,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 20,
  },
  statusText: {
    fontSize: 12,
    fontWeight: '600',
  },
  summaryCard: {
    backgroundColor: NEURAL_GRAY,
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: NEURAL_BORDER,
  },
  summaryLabel: {
    fontSize: 14,
    color: '#9BA1A6',
    marginBottom: 4,
  },
  summaryValue: {
    fontSize: 36,
    fontWeight: '700',
    color: '#FFFFFF',
    marginBottom: 8,
  },
  changeRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  changeText: {
    fontSize: 14,
    fontWeight: '600',
  },
  changeLabel: {
    fontSize: 12,
    color: '#6B7280',
  },
  statsRow: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 24,
  },
  statCard: {
    flex: 1,
    backgroundColor: NEURAL_GRAY,
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: NEURAL_BORDER,
  },
  statValue: {
    fontSize: 24,
    fontWeight: '700',
    color: '#FFFFFF',
    marginTop: 8,
  },
  statLabel: {
    fontSize: 11,
    color: '#6B7280',
    marginTop: 4,
    textAlign: 'center',
  },
  section: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#FFFFFF',
    marginBottom: 12,
  },
  watchlistItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: NEURAL_GRAY,
    borderRadius: 12,
    padding: 16,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: NEURAL_BORDER,
  },
  watchlistLeft: {
    flex: 1,
  },
  watchlistSymbol: {
    fontSize: 16,
    fontWeight: '700',
    color: '#FFFFFF',
  },
  watchlistName: {
    fontSize: 12,
    color: '#6B7280',
    marginTop: 2,
  },
  watchlistRight: {
    alignItems: 'flex-end',
  },
  watchlistPrice: {
    fontSize: 16,
    fontWeight: '600',
    color: '#FFFFFF',
  },
  changeBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
    marginTop: 4,
  },
  changePercent: {
    fontSize: 12,
    fontWeight: '600',
  },
});
