/**
 * Portfolio Screen
 * =================
 * Displays user's positions, P&L, and order history
 * Data from Layer 3 NestJS Backend
 */

import React, { useState } from 'react';
import { View, Text, StyleSheet, ScrollView, Pressable, FlatList } from 'react-native';
import Animated, { FadeIn, FadeInUp } from 'react-native-reanimated';
import { Wallet, TrendingUp, TrendingDown, Clock, DollarSign, PieChart, History } from 'lucide-react-native';

import { ScreenWrapper } from '@/components/common/screen-wrapper/screen-wrapper';
import { usePortfolioStore, selectSummary, selectPositions } from '@/store';
import { MATRIX_GREEN, STATUS_BEARISH, STATUS_NEUTRAL, NEURAL_GRAY, NEURAL_BORDER, NEURAL_BLACK } from '@/constants/theme';
import type { Position } from '@/store/portfolio.store';

// ============================================
// PORTFOLIO SCREEN
// ============================================

export default function PortfolioScreen() {
  const [activeTab, setActiveTab] = useState<'positions' | 'orders' | 'history'>('positions');
  
  const summary = usePortfolioStore(selectSummary);
  const positions = usePortfolioStore(selectPositions);

  // Demo data for initial display
  const demoSummary = {
    totalValue: 124567.89,
    totalCost: 100000,
    totalPnL: 24567.89,
    totalPnLPercent: 24.57,
    dayPnL: 2345.67,
    dayPnLPercent: 1.92,
    availableCash: 45678.90,
    marginUsed: 78889.00,
  };

  const demoPositions: Position[] = [
    {
      id: '1',
      symbol: 'AAPL',
      side: 'LONG',
      quantity: 100,
      entryPrice: 165.00,
      currentPrice: 178.45,
      unrealizedPnL: 1345.00,
      unrealizedPnLPercent: 8.15,
      openedAt: '2024-01-15T10:30:00Z',
    },
    {
      id: '2',
      symbol: 'TSLA',
      side: 'LONG',
      quantity: 50,
      entryPrice: 220.00,
      currentPrice: 242.68,
      unrealizedPnL: 1134.00,
      unrealizedPnLPercent: 10.31,
      openedAt: '2024-01-18T14:45:00Z',
    },
    {
      id: '3',
      symbol: 'GOOGL',
      side: 'SHORT',
      quantity: 25,
      entryPrice: 148.00,
      currentPrice: 141.23,
      unrealizedPnL: 169.25,
      unrealizedPnLPercent: 4.57,
      openedAt: '2024-01-20T09:15:00Z',
    },
    {
      id: '4',
      symbol: 'MSFT',
      side: 'LONG',
      quantity: 30,
      entryPrice: 405.00,
      currentPrice: 398.45,
      unrealizedPnL: -196.50,
      unrealizedPnLPercent: -1.62,
      openedAt: '2024-01-22T11:00:00Z',
    },
  ];

  const displaySummary = summary || demoSummary;
  const displayPositions = positions.length > 0 ? positions : demoPositions;

  const totalPositionValue = displayPositions.reduce(
    (sum, p) => sum + p.currentPrice * p.quantity,
    0
  );

  return (
    <ScreenWrapper>
      <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
        {/* Header */}
        <View style={styles.header}>
          <Wallet size={28} color={MATRIX_GREEN} />
          <Text style={styles.title}>Portfolio</Text>
        </View>

        {/* Summary Card */}
        <Animated.View entering={FadeIn} style={styles.summaryCard}>
          <Text style={styles.summaryLabel}>Total Portfolio Value</Text>
          <Text style={styles.summaryValue}>
            ${displaySummary.totalValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </Text>
          
          <View style={styles.pnlRow}>
            <View style={styles.pnlItem}>
              <Text style={styles.pnlLabel}>Total P&L</Text>
              <View style={styles.pnlValueRow}>
                {displaySummary.totalPnL >= 0 ? (
                  <TrendingUp size={16} color={MATRIX_GREEN} />
                ) : (
                  <TrendingDown size={16} color={STATUS_BEARISH} />
                )}
                <Text style={[
                  styles.pnlValue,
                  { color: displaySummary.totalPnL >= 0 ? MATRIX_GREEN : STATUS_BEARISH }
                ]}>
                  {displaySummary.totalPnL >= 0 ? '+' : ''}
                  ${displaySummary.totalPnL.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </Text>
              </View>
              <Text style={[
                styles.pnlPercent,
                { color: displaySummary.totalPnLPercent >= 0 ? MATRIX_GREEN : STATUS_BEARISH }
              ]}>
                ({displaySummary.totalPnLPercent >= 0 ? '+' : ''}{displaySummary.totalPnLPercent.toFixed(2)}%)
              </Text>
            </View>
            
            <View style={styles.pnlItem}>
              <Text style={styles.pnlLabel}>Today</Text>
              <View style={styles.pnlValueRow}>
                {displaySummary.dayPnL >= 0 ? (
                  <TrendingUp size={16} color={MATRIX_GREEN} />
                ) : (
                  <TrendingDown size={16} color={STATUS_BEARISH} />
                )}
                <Text style={[
                  styles.pnlValue,
                  { color: displaySummary.dayPnL >= 0 ? MATRIX_GREEN : STATUS_BEARISH }
                ]}>
                  {displaySummary.dayPnL >= 0 ? '+' : ''}
                  ${displaySummary.dayPnL.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </Text>
              </View>
              <Text style={[
                styles.pnlPercent,
                { color: displaySummary.dayPnLPercent >= 0 ? MATRIX_GREEN : STATUS_BEARISH }
              ]}>
                ({displaySummary.dayPnLPercent >= 0 ? '+' : ''}{displaySummary.dayPnLPercent.toFixed(2)}%)
              </Text>
            </View>
          </View>
        </Animated.View>

        {/* Stats Row */}
        <Animated.View entering={FadeIn.delay(100)} style={styles.statsRow}>
          <View style={styles.statCard}>
            <DollarSign size={18} color={MATRIX_GREEN} />
            <Text style={styles.statValue}>
              ${displaySummary.availableCash.toLocaleString(undefined, { maximumFractionDigits: 0 })}
            </Text>
            <Text style={styles.statLabel}>Available Cash</Text>
          </View>
          <View style={styles.statCard}>
            <PieChart size={18} color={STATUS_NEUTRAL} />
            <Text style={styles.statValue}>
              ${displaySummary.marginUsed.toLocaleString(undefined, { maximumFractionDigits: 0 })}
            </Text>
            <Text style={styles.statLabel}>Margin Used</Text>
          </View>
        </Animated.View>

        {/* Tab Navigation */}
        <View style={styles.tabContainer}>
          {[
            { key: 'positions', label: 'Positions', icon: TrendingUp },
            { key: 'orders', label: 'Orders', icon: Clock },
            { key: 'history', label: 'History', icon: History },
          ].map(({ key, label, icon: Icon }) => (
            <Pressable
              key={key}
              style={[
                styles.tab,
                activeTab === key && styles.tabActive,
              ]}
              onPress={() => setActiveTab(key as typeof activeTab)}
            >
              <Icon size={18} color={activeTab === key ? MATRIX_GREEN : '#6B7280'} />
              <Text style={[
                styles.tabText,
                activeTab === key && styles.tabTextActive,
              ]}>
                {label}
              </Text>
            </Pressable>
          ))}
        </View>

        {/* Positions List */}
        {activeTab === 'positions' && (
          <Animated.View entering={FadeInUp}>
            <Text style={styles.sectionTitle}>Open Positions ({displayPositions.length})</Text>
            {displayPositions.map((position, index) => (
              <PositionCard key={position.id} position={position} index={index} />
            ))}
          </Animated.View>
        )}

        {/* Orders Tab */}
        {activeTab === 'orders' && (
          <Animated.View entering={FadeInUp} style={styles.emptyState}>
            <Clock size={48} color="#6B7280" />
            <Text style={styles.emptyTitle}>No Pending Orders</Text>
            <Text style={styles.emptyText}>Your pending orders will appear here</Text>
          </Animated.View>
        )}

        {/* History Tab */}
        {activeTab === 'history' && (
          <Animated.View entering={FadeInUp} style={styles.emptyState}>
            <History size={48} color="#6B7280" />
            <Text style={styles.emptyTitle}>Trade History</Text>
            <Text style={styles.emptyText}>Your completed trades will appear here</Text>
          </Animated.View>
        )}
      </ScrollView>
    </ScreenWrapper>
  );
}

// ============================================
// POSITION CARD COMPONENT
// ============================================

function PositionCard({ position, index }: { position: Position; index: number }) {
  const isProfit = position.unrealizedPnL >= 0;
  const pnlColor = isProfit ? MATRIX_GREEN : STATUS_BEARISH;
  const sideColor = position.side === 'LONG' ? MATRIX_GREEN : STATUS_BEARISH;

  return (
    <Animated.View
      entering={FadeInUp.delay(index * 50)}
      style={styles.positionCard}
    >
      <View style={styles.positionHeader}>
        <View style={styles.positionLeft}>
          <Text style={styles.positionSymbol}>{position.symbol}</Text>
          <View style={[styles.sideBadge, { backgroundColor: sideColor + '20' }]}>
            <Text style={[styles.sideText, { color: sideColor }]}>{position.side}</Text>
          </View>
        </View>
        <View style={styles.positionRight}>
          <Text style={[styles.positionPnL, { color: pnlColor }]}>
            {isProfit ? '+' : ''}${position.unrealizedPnL.toFixed(2)}
          </Text>
          <Text style={[styles.positionPnLPercent, { color: pnlColor }]}>
            ({isProfit ? '+' : ''}{position.unrealizedPnLPercent.toFixed(2)}%)
          </Text>
        </View>
      </View>

      <View style={styles.positionDetails}>
        <View style={styles.detailItem}>
          <Text style={styles.detailLabel}>Qty</Text>
          <Text style={styles.detailValue}>{position.quantity}</Text>
        </View>
        <View style={styles.detailItem}>
          <Text style={styles.detailLabel}>Entry</Text>
          <Text style={styles.detailValue}>${position.entryPrice.toFixed(2)}</Text>
        </View>
        <View style={styles.detailItem}>
          <Text style={styles.detailLabel}>Current</Text>
          <Text style={styles.detailValue}>${position.currentPrice.toFixed(2)}</Text>
        </View>
        <View style={styles.detailItem}>
          <Text style={styles.detailLabel}>Value</Text>
          <Text style={styles.detailValue}>
            ${(position.quantity * position.currentPrice).toLocaleString(undefined, { maximumFractionDigits: 0 })}
          </Text>
        </View>
      </View>
    </Animated.View>
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
    alignItems: 'center',
    gap: 12,
    marginBottom: 20,
    marginTop: 8,
  },
  title: {
    fontSize: 28,
    fontWeight: '700',
    color: '#FFFFFF',
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
    marginBottom: 16,
  },
  pnlRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  pnlItem: {
    flex: 1,
  },
  pnlLabel: {
    fontSize: 12,
    color: '#6B7280',
    marginBottom: 4,
  },
  pnlValueRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  pnlValue: {
    fontSize: 18,
    fontWeight: '700',
  },
  pnlPercent: {
    fontSize: 12,
    fontWeight: '600',
    marginTop: 2,
  },
  statsRow: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 20,
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
    fontSize: 18,
    fontWeight: '700',
    color: '#FFFFFF',
    marginTop: 8,
  },
  statLabel: {
    fontSize: 11,
    color: '#6B7280',
    marginTop: 4,
  },
  tabContainer: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: 20,
  },
  tab: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    paddingVertical: 12,
    borderRadius: 10,
    backgroundColor: NEURAL_GRAY,
    borderWidth: 1,
    borderColor: NEURAL_BORDER,
  },
  tabActive: {
    backgroundColor: MATRIX_GREEN + '20',
    borderColor: MATRIX_GREEN,
  },
  tabText: {
    fontSize: 13,
    fontWeight: '600',
    color: '#6B7280',
  },
  tabTextActive: {
    color: MATRIX_GREEN,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#FFFFFF',
    marginBottom: 12,
  },
  positionCard: {
    backgroundColor: NEURAL_GRAY,
    borderRadius: 12,
    padding: 16,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: NEURAL_BORDER,
  },
  positionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  positionLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  positionSymbol: {
    fontSize: 18,
    fontWeight: '700',
    color: '#FFFFFF',
  },
  sideBadge: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 4,
  },
  sideText: {
    fontSize: 10,
    fontWeight: '700',
  },
  positionRight: {
    alignItems: 'flex-end',
  },
  positionPnL: {
    fontSize: 16,
    fontWeight: '700',
  },
  positionPnLPercent: {
    fontSize: 12,
    fontWeight: '600',
  },
  positionDetails: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  detailItem: {
    alignItems: 'center',
  },
  detailLabel: {
    fontSize: 10,
    color: '#6B7280',
    marginBottom: 2,
  },
  detailValue: {
    fontSize: 13,
    fontWeight: '600',
    color: '#FFFFFF',
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 48,
  },
  emptyTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#FFFFFF',
    marginTop: 16,
  },
  emptyText: {
    fontSize: 14,
    color: '#6B7280',
    marginTop: 8,
  },
});
