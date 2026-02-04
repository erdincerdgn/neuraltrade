/**
 * AI Signals Screen
 * ==================
 * Displays AI-generated trading signals from Layer 4 Python AI Engine
 * Real-time updates via WebSocket + Agent terminal view
 */

import React, { useState } from 'react';
import { View, Text, StyleSheet, FlatList, Pressable, ScrollView } from 'react-native';
import Animated, { FadeIn, FadeInUp } from 'react-native-reanimated';
import { Brain, Filter, Terminal, Zap, TrendingUp, TrendingDown, Minus } from 'lucide-react-native';

import { ScreenWrapper } from '@/components/common/screen-wrapper/screen-wrapper';
import { SignalCard } from '@/components/common/signal-card/signal-card';
import { useSocket } from '@/hooks/use-socket';
import { useSignalsStore, selectFilteredSignals, selectAgentThoughts } from '@/store';
import { MATRIX_GREEN, STATUS_BEARISH, STATUS_NEUTRAL, NEURAL_GRAY, NEURAL_BORDER, NEURAL_BLACK } from '@/constants/theme';
import type { AISignal } from '@/services/socket';

// ============================================
// AI SIGNALS SCREEN
// ============================================

export default function AISignalsScreen() {
  const [showTerminal, setShowTerminal] = useState(false);
  const [activeFilter, setActiveFilter] = useState<'ALL' | 'BUY' | 'SELL' | 'HOLD'>('ALL');
  
  // Connect to socket for signals
  useSocket({ autoConnect: true, enableSignals: true, enableAgentThoughts: showTerminal });
  
  const signals = useSignalsStore(selectFilteredSignals);
  const agentThoughts = useSignalsStore(selectAgentThoughts);
  const setFilterAction = useSignalsStore((state) => state.setFilterAction);

  // Demo signals for initial display
  const demoSignals: AISignal[] = [
    {
      id: '1',
      symbol: 'AAPL',
      action: 'BUY',
      confidence: 0.87,
      price: 178.45,
      stopLoss: 172.00,
      takeProfit: 195.00,
      reasoning: 'Strong bullish momentum with RSI breakout. ICT Silver Bullet pattern detected at 10:00 AM session.',
      agentId: 'swarm-alpha',
      timestamp: Date.now() - 300000,
    },
    {
      id: '2',
      symbol: 'TSLA',
      action: 'SELL',
      confidence: 0.72,
      price: 242.68,
      stopLoss: 255.00,
      takeProfit: 220.00,
      reasoning: 'Bearish divergence on 4H chart. Fair value gap filled, expecting retracement to 220 zone.',
      agentId: 'swarm-beta',
      timestamp: Date.now() - 600000,
    },
    {
      id: '3',
      symbol: 'GOOGL',
      action: 'HOLD',
      confidence: 0.65,
      price: 141.23,
      reasoning: 'Consolidating in range. Wait for breakout confirmation above 145 or breakdown below 138.',
      agentId: 'swarm-gamma',
      timestamp: Date.now() - 900000,
    },
  ];

  const displaySignals = signals.length > 0 ? signals : demoSignals;

  const handleFilterChange = (filter: 'ALL' | 'BUY' | 'SELL' | 'HOLD') => {
    setActiveFilter(filter);
    setFilterAction(filter);
  };

  return (
    <ScreenWrapper>
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <Brain size={28} color={MATRIX_GREEN} />
          <Text style={styles.title}>AI Signals</Text>
        </View>
        <Pressable
          style={[styles.terminalButton, showTerminal && styles.terminalButtonActive]}
          onPress={() => setShowTerminal(!showTerminal)}
        >
          <Terminal size={20} color={showTerminal ? NEURAL_BLACK : MATRIX_GREEN} />
        </Pressable>
      </View>

      {/* Stats Bar */}
      <Animated.View entering={FadeIn} style={styles.statsBar}>
        <View style={styles.stat}>
          <TrendingUp size={16} color={MATRIX_GREEN} />
          <Text style={styles.statValue}>5</Text>
          <Text style={styles.statLabel}>Buy</Text>
        </View>
        <View style={styles.stat}>
          <TrendingDown size={16} color={STATUS_BEARISH} />
          <Text style={styles.statValue}>3</Text>
          <Text style={styles.statLabel}>Sell</Text>
        </View>
        <View style={styles.stat}>
          <Minus size={16} color={STATUS_NEUTRAL} />
          <Text style={styles.statValue}>4</Text>
          <Text style={styles.statLabel}>Hold</Text>
        </View>
        <View style={styles.stat}>
          <Zap size={16} color={MATRIX_GREEN} />
          <Text style={styles.statValue}>78%</Text>
          <Text style={styles.statLabel}>Accuracy</Text>
        </View>
      </Animated.View>

      {/* Filter Tabs */}
      <View style={styles.filterContainer}>
        {(['ALL', 'BUY', 'SELL', 'HOLD'] as const).map((filter) => (
          <Pressable
            key={filter}
            style={[
              styles.filterTab,
              activeFilter === filter && styles.filterTabActive,
            ]}
            onPress={() => handleFilterChange(filter)}
          >
            <Text
              style={[
                styles.filterText,
                activeFilter === filter && styles.filterTextActive,
              ]}
            >
              {filter}
            </Text>
          </Pressable>
        ))}
      </View>

      {/* Agent Terminal View */}
      {showTerminal && (
        <Animated.View entering={FadeInUp} style={styles.terminal}>
          <View style={styles.terminalHeader}>
            <Terminal size={14} color={MATRIX_GREEN} />
            <Text style={styles.terminalTitle}>Agent Swarm Terminal</Text>
          </View>
          <ScrollView style={styles.terminalContent}>
            {agentThoughts.length === 0 ? (
              <>
                <Text style={styles.terminalLine}>
                  <Text style={styles.terminalAgent}>[swarm-alpha]</Text> Analyzing AAPL price action...
                </Text>
                <Text style={styles.terminalLine}>
                  <Text style={styles.terminalAgent}>[swarm-beta]</Text> Checking RSI divergence patterns...
                </Text>
                <Text style={styles.terminalLine}>
                  <Text style={styles.terminalAgent}>[swarm-gamma]</Text> Scanning for FVG opportunities...
                </Text>
                <Text style={styles.terminalLine}>
                  <Text style={styles.terminalAgent}>[orchestrator]</Text> Consensus reached: BUY AAPL @ 178.45
                </Text>
              </>
            ) : (
              agentThoughts.map((thought, index) => (
                <Text key={index} style={styles.terminalLine}>
                  <Text style={styles.terminalAgent}>[{thought.agentName}]</Text> {thought.thought}
                </Text>
              ))
            )}
          </ScrollView>
        </Animated.View>
      )}

      {/* Signals List */}
      <FlatList
        data={displaySignals}
        keyExtractor={(item) => item.id}
        renderItem={({ item, index }) => (
          <SignalCard signal={item} index={index} />
        )}
        contentContainerStyle={styles.listContent}
        showsVerticalScrollIndicator={false}
      />
    </ScreenWrapper>
  );
}

// ============================================
// STYLES
// ============================================

const styles = StyleSheet.create({
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
    marginTop: 8,
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  title: {
    fontSize: 28,
    fontWeight: '700',
    color: '#FFFFFF',
  },
  terminalButton: {
    padding: 10,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: MATRIX_GREEN,
  },
  terminalButtonActive: {
    backgroundColor: MATRIX_GREEN,
  },
  statsBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    backgroundColor: NEURAL_GRAY,
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: NEURAL_BORDER,
  },
  stat: {
    alignItems: 'center',
    gap: 4,
  },
  statValue: {
    fontSize: 18,
    fontWeight: '700',
    color: '#FFFFFF',
  },
  statLabel: {
    fontSize: 11,
    color: '#6B7280',
  },
  filterContainer: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: 16,
  },
  filterTab: {
    flex: 1,
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderRadius: 8,
    backgroundColor: NEURAL_GRAY,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: NEURAL_BORDER,
  },
  filterTabActive: {
    backgroundColor: MATRIX_GREEN + '20',
    borderColor: MATRIX_GREEN,
  },
  filterText: {
    fontSize: 13,
    fontWeight: '600',
    color: '#6B7280',
  },
  filterTextActive: {
    color: MATRIX_GREEN,
  },
  terminal: {
    backgroundColor: NEURAL_BLACK,
    borderRadius: 12,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: MATRIX_GREEN + '40',
    overflow: 'hidden',
  },
  terminalHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    padding: 12,
    backgroundColor: NEURAL_GRAY,
    borderBottomWidth: 1,
    borderBottomColor: NEURAL_BORDER,
  },
  terminalTitle: {
    fontSize: 12,
    fontWeight: '600',
    color: MATRIX_GREEN,
    fontFamily: 'monospace',
  },
  terminalContent: {
    padding: 12,
    maxHeight: 120,
  },
  terminalLine: {
    fontSize: 11,
    color: '#9BA1A6',
    fontFamily: 'monospace',
    marginBottom: 6,
    lineHeight: 16,
  },
  terminalAgent: {
    color: MATRIX_GREEN,
    fontWeight: '600',
  },
  listContent: {
    paddingBottom: 24,
  },
});
