/**
 * Tab Layout
 * ===========
 * NeuralTrade Bottom Navigation
 * 
 * Screens:
 * - Dashboard: Market overview & watchlist
 * - AI Signals: AI-generated trading signals
 * - Trade: Order execution
 * - Portfolio: Positions & P&L
 * - Profile: User settings & logout
 */

import { Tabs } from 'expo-router';
import React from 'react';
import { StyleSheet, View } from 'react-native';
import { LayoutDashboard, Brain, ArrowUpDown, Wallet, User } from 'lucide-react-native';

import { HapticTab } from '@/components/common/habtic-tabs/haptic-tab';
import { NEURAL_BLACK, NEURAL_GRAY, MATRIX_GREEN, NEURAL_BORDER } from '@/constants/theme';

export default function TabLayout() {
  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarActiveTintColor: MATRIX_GREEN,
        tabBarInactiveTintColor: '#6B7280',
        tabBarStyle: styles.tabBar,
        tabBarLabelStyle: styles.tabBarLabel,
        tabBarButton: HapticTab,
      }}
    >
      <Tabs.Screen
        name="index"
        options={{
          title: 'Dashboard',
          tabBarIcon: ({ color, focused }) => (
            <View style={focused && styles.activeIconContainer}>
              <LayoutDashboard size={24} color={color} />
            </View>
          ),
        }}
      />
      <Tabs.Screen
        name="ai-signals"
        options={{
          title: 'Signals',
          tabBarIcon: ({ color, focused }) => (
            <View style={focused && styles.activeIconContainer}>
              <Brain size={24} color={color} />
            </View>
          ),
        }}
      />
      <Tabs.Screen
        name="trade"
        options={{
          title: 'Trade',
          tabBarIcon: ({ color, focused }) => (
            <View style={focused && styles.activeIconContainer}>
              <ArrowUpDown size={24} color={color} />
            </View>
          ),
        }}
      />
      <Tabs.Screen
        name="portfolio"
        options={{
          title: 'Portfolio',
          tabBarIcon: ({ color, focused }) => (
            <View style={focused && styles.activeIconContainer}>
              <Wallet size={24} color={color} />
            </View>
          ),
        }}
      />
      <Tabs.Screen
        name="profile"
        options={{
          title: 'Profile',
          tabBarIcon: ({ color, focused }) => (
            <View style={focused && styles.activeIconContainer}>
              <User size={24} color={color} />
            </View>
          ),
        }}
      />
      {/* Hide explore from tabs but keep for routing */}
      <Tabs.Screen
        name="explore"
        options={{
          href: null,
        }}
      />
    </Tabs>
  );
}

const styles = StyleSheet.create({
  tabBar: {
    backgroundColor: NEURAL_GRAY,
    borderTopColor: NEURAL_BORDER,
    borderTopWidth: 1,
    height: 70,
    paddingBottom: 8,
    paddingTop: 8,
  },
  tabBarLabel: {
    fontSize: 11,
    fontWeight: '600',
  },
  activeIconContainer: {
    backgroundColor: MATRIX_GREEN + '20',
    padding: 6,
    borderRadius: 8,
  },
});
