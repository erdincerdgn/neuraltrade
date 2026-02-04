/**
 * Trade Screen
 * ==============
 * Order execution screen with chart view
 * Connects to Layer 3 for order management
 */

import React, { useState } from 'react';
import { View, Text, StyleSheet, TextInput, Pressable, ScrollView } from 'react-native';
import Animated, { FadeIn, FadeInUp } from 'react-native-reanimated';
import { ArrowUpDown, TrendingUp, TrendingDown, ChevronDown, AlertCircle } from 'lucide-react-native';

import { ScreenWrapper } from '@/components/common/screen-wrapper/screen-wrapper';
import { MATRIX_GREEN, STATUS_BEARISH, STATUS_NEUTRAL, NEURAL_GRAY, NEURAL_BORDER, NEURAL_BLACK } from '@/constants/theme';

// ============================================
// TYPES
// ============================================

type OrderSide = 'BUY' | 'SELL';
type OrderType = 'MARKET' | 'LIMIT' | 'STOP';

// ============================================
// TRADE SCREEN
// ============================================

export default function TradeScreen() {
  const [symbol, setSymbol] = useState('AAPL');
  const [orderSide, setOrderSide] = useState<OrderSide>('BUY');
  const [orderType, setOrderType] = useState<OrderType>('MARKET');
  const [quantity, setQuantity] = useState('10');
  const [price, setPrice] = useState('178.45');
  const [stopLoss, setStopLoss] = useState('');
  const [takeProfit, setTakeProfit] = useState('');

  const currentPrice = 178.45;
  const estimatedCost = parseFloat(quantity || '0') * currentPrice;

  const handlePlaceOrder = () => {
    // TODO: Integrate with API
    console.log('Placing order:', {
      symbol,
      side: orderSide,
      type: orderType,
      quantity: parseFloat(quantity),
      price: orderType !== 'MARKET' ? parseFloat(price) : undefined,
      stopLoss: stopLoss ? parseFloat(stopLoss) : undefined,
      takeProfit: takeProfit ? parseFloat(takeProfit) : undefined,
    });
  };

  return (
    <ScreenWrapper>
      <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
        {/* Header */}
        <View style={styles.header}>
          <ArrowUpDown size={28} color={MATRIX_GREEN} />
          <Text style={styles.title}>Trade</Text>
        </View>

        {/* Symbol Selector */}
        <Animated.View entering={FadeIn} style={styles.symbolCard}>
          <View style={styles.symbolHeader}>
            <View>
              <Text style={styles.symbolName}>{symbol}</Text>
              <Text style={styles.symbolFullName}>Apple Inc.</Text>
            </View>
            <View style={styles.priceContainer}>
              <Text style={styles.currentPrice}>${currentPrice.toFixed(2)}</Text>
              <View style={styles.changeContainer}>
                <TrendingUp size={14} color={MATRIX_GREEN} />
                <Text style={[styles.change, { color: MATRIX_GREEN }]}>+2.34 (+1.33%)</Text>
              </View>
            </View>
          </View>
          
          {/* Mini Chart Placeholder */}
          <View style={styles.chartPlaceholder}>
            <Text style={styles.chartPlaceholderText}>📈 TradingView Chart (WebView)</Text>
          </View>
        </Animated.View>

        {/* Order Side Toggle */}
        <Animated.View entering={FadeInUp.delay(100)} style={styles.sideToggle}>
          <Pressable
            style={[
              styles.sideButton,
              orderSide === 'BUY' && styles.sideButtonBuy,
            ]}
            onPress={() => setOrderSide('BUY')}
          >
            <TrendingUp size={20} color={orderSide === 'BUY' ? NEURAL_BLACK : '#6B7280'} />
            <Text style={[
              styles.sideButtonText,
              orderSide === 'BUY' && styles.sideButtonTextActive,
            ]}>
              BUY
            </Text>
          </Pressable>
          <Pressable
            style={[
              styles.sideButton,
              orderSide === 'SELL' && styles.sideButtonSell,
            ]}
            onPress={() => setOrderSide('SELL')}
          >
            <TrendingDown size={20} color={orderSide === 'SELL' ? NEURAL_BLACK : '#6B7280'} />
            <Text style={[
              styles.sideButtonText,
              orderSide === 'SELL' && styles.sideButtonTextActive,
            ]}>
              SELL
            </Text>
          </Pressable>
        </Animated.View>

        {/* Order Type Selector */}
        <Animated.View entering={FadeInUp.delay(150)} style={styles.section}>
          <Text style={styles.label}>Order Type</Text>
          <View style={styles.typeContainer}>
            {(['MARKET', 'LIMIT', 'STOP'] as OrderType[]).map((type) => (
              <Pressable
                key={type}
                style={[
                  styles.typeButton,
                  orderType === type && styles.typeButtonActive,
                ]}
                onPress={() => setOrderType(type)}
              >
                <Text style={[
                  styles.typeButtonText,
                  orderType === type && styles.typeButtonTextActive,
                ]}>
                  {type}
                </Text>
              </Pressable>
            ))}
          </View>
        </Animated.View>

        {/* Quantity Input */}
        <Animated.View entering={FadeInUp.delay(200)} style={styles.section}>
          <Text style={styles.label}>Quantity</Text>
          <View style={styles.inputContainer}>
            <TextInput
              style={styles.input}
              value={quantity}
              onChangeText={setQuantity}
              keyboardType="numeric"
              placeholder="0"
              placeholderTextColor="#6B7280"
            />
            <Text style={styles.inputSuffix}>shares</Text>
          </View>
        </Animated.View>

        {/* Price Input (for Limit/Stop orders) */}
        {orderType !== 'MARKET' && (
          <Animated.View entering={FadeInUp.delay(250)} style={styles.section}>
            <Text style={styles.label}>{orderType === 'LIMIT' ? 'Limit Price' : 'Stop Price'}</Text>
            <View style={styles.inputContainer}>
              <Text style={styles.inputPrefix}>$</Text>
              <TextInput
                style={styles.input}
                value={price}
                onChangeText={setPrice}
                keyboardType="decimal-pad"
                placeholder="0.00"
                placeholderTextColor="#6B7280"
              />
            </View>
          </Animated.View>
        )}

        {/* Stop Loss & Take Profit */}
        <Animated.View entering={FadeInUp.delay(300)} style={styles.riskRow}>
          <View style={[styles.section, { flex: 1 }]}>
            <Text style={styles.label}>Stop Loss</Text>
            <View style={styles.inputContainer}>
              <Text style={styles.inputPrefix}>$</Text>
              <TextInput
                style={styles.input}
                value={stopLoss}
                onChangeText={setStopLoss}
                keyboardType="decimal-pad"
                placeholder="Optional"
                placeholderTextColor="#6B7280"
              />
            </View>
          </View>
          <View style={[styles.section, { flex: 1 }]}>
            <Text style={styles.label}>Take Profit</Text>
            <View style={styles.inputContainer}>
              <Text style={styles.inputPrefix}>$</Text>
              <TextInput
                style={styles.input}
                value={takeProfit}
                onChangeText={setTakeProfit}
                keyboardType="decimal-pad"
                placeholder="Optional"
                placeholderTextColor="#6B7280"
              />
            </View>
          </View>
        </Animated.View>

        {/* Order Summary */}
        <Animated.View entering={FadeInUp.delay(350)} style={styles.summaryCard}>
          <View style={styles.summaryRow}>
            <Text style={styles.summaryLabel}>Estimated Cost</Text>
            <Text style={styles.summaryValue}>${estimatedCost.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</Text>
          </View>
          <View style={styles.summaryRow}>
            <Text style={styles.summaryLabel}>Commission</Text>
            <Text style={styles.summaryValue}>$0.00</Text>
          </View>
          <View style={[styles.summaryRow, styles.summaryRowTotal]}>
            <Text style={styles.summaryLabelTotal}>Total</Text>
            <Text style={styles.summaryValueTotal}>${estimatedCost.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</Text>
          </View>
        </Animated.View>

        {/* Place Order Button */}
        <Animated.View entering={FadeInUp.delay(400)}>
          <Pressable
            style={[
              styles.orderButton,
              orderSide === 'BUY' ? styles.orderButtonBuy : styles.orderButtonSell,
            ]}
            onPress={handlePlaceOrder}
          >
            <Text style={styles.orderButtonText}>
              {orderSide} {quantity} {symbol}
            </Text>
          </Pressable>
        </Animated.View>

        {/* Risk Warning */}
        <View style={styles.warning}>
          <AlertCircle size={16} color={STATUS_NEUTRAL} />
          <Text style={styles.warningText}>
            Trading involves risk. Please review your order carefully.
          </Text>
        </View>
      </ScrollView>
    </ScreenWrapper>
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
  symbolCard: {
    backgroundColor: NEURAL_GRAY,
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: NEURAL_BORDER,
  },
  symbolHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  symbolName: {
    fontSize: 24,
    fontWeight: '700',
    color: '#FFFFFF',
  },
  symbolFullName: {
    fontSize: 12,
    color: '#6B7280',
    marginTop: 2,
  },
  priceContainer: {
    alignItems: 'flex-end',
  },
  currentPrice: {
    fontSize: 24,
    fontWeight: '700',
    color: '#FFFFFF',
  },
  changeContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    marginTop: 4,
  },
  change: {
    fontSize: 12,
    fontWeight: '600',
  },
  chartPlaceholder: {
    height: 100,
    backgroundColor: NEURAL_BLACK,
    borderRadius: 8,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: NEURAL_BORDER,
  },
  chartPlaceholderText: {
    color: '#6B7280',
    fontSize: 12,
  },
  sideToggle: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 20,
  },
  sideButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 16,
    borderRadius: 12,
    backgroundColor: NEURAL_GRAY,
    borderWidth: 1,
    borderColor: NEURAL_BORDER,
  },
  sideButtonBuy: {
    backgroundColor: MATRIX_GREEN,
    borderColor: MATRIX_GREEN,
  },
  sideButtonSell: {
    backgroundColor: STATUS_BEARISH,
    borderColor: STATUS_BEARISH,
  },
  sideButtonText: {
    fontSize: 16,
    fontWeight: '700',
    color: '#6B7280',
  },
  sideButtonTextActive: {
    color: NEURAL_BLACK,
  },
  section: {
    marginBottom: 16,
  },
  label: {
    fontSize: 13,
    fontWeight: '600',
    color: '#9BA1A6',
    marginBottom: 8,
  },
  typeContainer: {
    flexDirection: 'row',
    gap: 8,
  },
  typeButton: {
    flex: 1,
    paddingVertical: 12,
    borderRadius: 8,
    backgroundColor: NEURAL_GRAY,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: NEURAL_BORDER,
  },
  typeButtonActive: {
    backgroundColor: MATRIX_GREEN + '20',
    borderColor: MATRIX_GREEN,
  },
  typeButtonText: {
    fontSize: 13,
    fontWeight: '600',
    color: '#6B7280',
  },
  typeButtonTextActive: {
    color: MATRIX_GREEN,
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: NEURAL_GRAY,
    borderRadius: 12,
    paddingHorizontal: 16,
    borderWidth: 1,
    borderColor: NEURAL_BORDER,
  },
  inputPrefix: {
    fontSize: 18,
    color: '#6B7280',
    marginRight: 4,
  },
  input: {
    flex: 1,
    fontSize: 18,
    fontWeight: '600',
    color: '#FFFFFF',
    paddingVertical: 14,
  },
  inputSuffix: {
    fontSize: 14,
    color: '#6B7280',
  },
  riskRow: {
    flexDirection: 'row',
    gap: 12,
  },
  summaryCard: {
    backgroundColor: NEURAL_GRAY,
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: NEURAL_BORDER,
  },
  summaryRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  summaryRowTotal: {
    marginTop: 8,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: NEURAL_BORDER,
    marginBottom: 0,
  },
  summaryLabel: {
    fontSize: 14,
    color: '#6B7280',
  },
  summaryValue: {
    fontSize: 14,
    color: '#FFFFFF',
  },
  summaryLabelTotal: {
    fontSize: 16,
    fontWeight: '600',
    color: '#FFFFFF',
  },
  summaryValueTotal: {
    fontSize: 18,
    fontWeight: '700',
    color: MATRIX_GREEN,
  },
  orderButton: {
    paddingVertical: 18,
    borderRadius: 12,
    alignItems: 'center',
    marginBottom: 12,
  },
  orderButtonBuy: {
    backgroundColor: MATRIX_GREEN,
  },
  orderButtonSell: {
    backgroundColor: STATUS_BEARISH,
  },
  orderButtonText: {
    fontSize: 18,
    fontWeight: '700',
    color: NEURAL_BLACK,
  },
  warning: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    padding: 12,
    backgroundColor: STATUS_NEUTRAL + '15',
    borderRadius: 8,
    marginBottom: 24,
  },
  warningText: {
    flex: 1,
    fontSize: 12,
    color: STATUS_NEUTRAL,
  },
});
