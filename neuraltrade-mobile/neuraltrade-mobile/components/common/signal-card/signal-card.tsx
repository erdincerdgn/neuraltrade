/**
 * Signal Card Component
 * ======================
 * Displays AI trading signal with animated entry
 */

import React from 'react';
import { View, Text, StyleSheet, Pressable } from 'react-native';
import Animated, { FadeInUp } from 'react-native-reanimated';
import { TrendingUp, TrendingDown, Minus, Bot } from 'lucide-react-native';
import { MATRIX_GREEN, STATUS_BEARISH, STATUS_NEUTRAL, NEURAL_GRAY, NEURAL_BORDER } from '@/constants/theme';
import type { AISignal } from '@/services/socket';

interface SignalCardProps {
  signal: AISignal;
  onPress?: () => void;
  index?: number;
}

const AnimatedPressable = Animated.createAnimatedComponent(Pressable);

export function SignalCard({ signal, onPress, index = 0 }: SignalCardProps) {
  const getActionColor = () => {
    switch (signal.action) {
      case 'BUY':
        return MATRIX_GREEN;
      case 'SELL':
        return STATUS_BEARISH;
      default:
        return STATUS_NEUTRAL;
    }
  };

  const getActionIcon = () => {
    switch (signal.action) {
      case 'BUY':
        return <TrendingUp size={20} color={MATRIX_GREEN} />;
      case 'SELL':
        return <TrendingDown size={20} color={STATUS_BEARISH} />;
      default:
        return <Minus size={20} color={STATUS_NEUTRAL} />;
    }
  };

  const confidencePercent = Math.round(signal.confidence * 100);
  const actionColor = getActionColor();

  return (
    <AnimatedPressable
      entering={FadeInUp.delay(index * 100).springify()}
      onPress={onPress}
      style={styles.container}
    >
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.symbolContainer}>
          <Text style={styles.symbol}>{signal.symbol}</Text>
          <View style={[styles.actionBadge, { backgroundColor: actionColor + '20' }]}>
            {getActionIcon()}
            <Text style={[styles.actionText, { color: actionColor }]}>
              {signal.action}
            </Text>
          </View>
        </View>
        <View style={styles.confidenceContainer}>
          <Text style={styles.confidenceLabel}>Confidence</Text>
          <Text style={[styles.confidenceValue, { color: actionColor }]}>
            {confidencePercent}%
          </Text>
        </View>
      </View>

      {/* Price Info */}
      <View style={styles.priceRow}>
        <View style={styles.priceItem}>
          <Text style={styles.priceLabel}>Entry</Text>
          <Text style={styles.priceValue}>${signal.price.toFixed(2)}</Text>
        </View>
        {signal.stopLoss && (
          <View style={styles.priceItem}>
            <Text style={styles.priceLabel}>Stop Loss</Text>
            <Text style={[styles.priceValue, { color: STATUS_BEARISH }]}>
              ${signal.stopLoss.toFixed(2)}
            </Text>
          </View>
        )}
        {signal.takeProfit && (
          <View style={styles.priceItem}>
            <Text style={styles.priceLabel}>Take Profit</Text>
            <Text style={[styles.priceValue, { color: MATRIX_GREEN }]}>
              ${signal.takeProfit.toFixed(2)}
            </Text>
          </View>
        )}
      </View>

      {/* AI Reasoning */}
      <View style={styles.reasoningContainer}>
        <Bot size={14} color="#6B7280" />
        <Text style={styles.reasoning} numberOfLines={2}>
          {signal.reasoning}
        </Text>
      </View>

      {/* Timestamp */}
      <Text style={styles.timestamp}>
        {new Date(signal.timestamp).toLocaleTimeString()}
      </Text>
    </AnimatedPressable>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: NEURAL_GRAY,
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: NEURAL_BORDER,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  symbolContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  symbol: {
    fontSize: 18,
    fontWeight: '700',
    color: '#FFFFFF',
  },
  actionBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
    gap: 4,
  },
  actionText: {
    fontSize: 12,
    fontWeight: '600',
  },
  confidenceContainer: {
    alignItems: 'flex-end',
  },
  confidenceLabel: {
    fontSize: 11,
    color: '#6B7280',
  },
  confidenceValue: {
    fontSize: 20,
    fontWeight: '700',
  },
  priceRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  priceItem: {
    flex: 1,
  },
  priceLabel: {
    fontSize: 11,
    color: '#6B7280',
    marginBottom: 2,
  },
  priceValue: {
    fontSize: 14,
    fontWeight: '600',
    color: '#FFFFFF',
  },
  reasoningContainer: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 6,
    backgroundColor: '#0A0A0A',
    padding: 10,
    borderRadius: 8,
    marginBottom: 8,
  },
  reasoning: {
    flex: 1,
    fontSize: 12,
    color: '#9BA1A6',
    lineHeight: 18,
  },
  timestamp: {
    fontSize: 11,
    color: '#4B5563',
    textAlign: 'right',
  },
});

export default SignalCard;
