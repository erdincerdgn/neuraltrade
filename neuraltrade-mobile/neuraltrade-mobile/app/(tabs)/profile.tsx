/**
 * Profile Screen
 * ===============
 * User profile management and settings
 * Includes logout functionality
 */

import React from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  Alert,
  Image,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import {
  User,
  Settings,
  Bell,
  Shield,
  HelpCircle,
  LogOut,
  ChevronRight,
  Wallet,
  TrendingUp,
} from 'lucide-react-native';
import { useAuthStore } from '@/store/auth.store';
import { authService } from '@/services/auth';

// ============================================
// THEME COLORS
// ============================================
const COLORS = {
  background: '#000000',
  surface: '#0D0D0D',
  border: '#1A1A1A',
  primary: '#00FF41',
  primaryDim: 'rgba(0, 255, 65, 0.1)',
  text: '#FFFFFF',
  textSecondary: '#888888',
  danger: '#FF3B30',
  dangerDim: 'rgba(255, 59, 48, 0.1)',
};

// ============================================
// MENU ITEM COMPONENT
// ============================================
interface MenuItemProps {
  icon: React.ComponentType<any>;
  title: string;
  subtitle?: string;
  onPress: () => void;
  danger?: boolean;
  showChevron?: boolean;
}

function MenuItem({ icon: Icon, title, subtitle, onPress, danger, showChevron = true }: MenuItemProps) {
  return (
    <TouchableOpacity
      style={{
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: danger ? COLORS.dangerDim : COLORS.surface,
        padding: 16,
        borderRadius: 12,
        marginBottom: 8,
      }}
      onPress={onPress}
    >
      <View style={{
        width: 40,
        height: 40,
        borderRadius: 10,
        backgroundColor: danger ? COLORS.dangerDim : COLORS.primaryDim,
        justifyContent: 'center',
        alignItems: 'center',
        marginRight: 12,
      }}>
        <Icon size={20} color={danger ? COLORS.danger : COLORS.primary} />
      </View>
      <View style={{ flex: 1 }}>
        <Text style={{ 
          color: danger ? COLORS.danger : COLORS.text, 
          fontSize: 16, 
          fontWeight: '500' 
        }}>
          {title}
        </Text>
        {subtitle && (
          <Text style={{ color: COLORS.textSecondary, fontSize: 13, marginTop: 2 }}>
            {subtitle}
          </Text>
        )}
      </View>
      {showChevron && !danger && <ChevronRight size={20} color={COLORS.textSecondary} />}
    </TouchableOpacity>
  );
}

// ============================================
// PROFILE SCREEN
// ============================================
export default function ProfileScreen() {
  const router = useRouter();
  const { user, subscription } = useAuthStore();

  const handleLogout = () => {
    Alert.alert(
      'Sign Out',
      'Are you sure you want to sign out?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Sign Out',
          style: 'destructive',
          onPress: async () => {
            await authService.logout();
            router.replace('/login' as any);
          },
        },
      ]
    );
  };

  const getRiskProfileColor = (profile?: string) => {
    switch (profile) {
      case 'LOW': return '#4CAF50';
      case 'MODERATE': return '#FFC107';
      case 'HIGH': return '#FF9800';
      case 'AGGRESSIVE': return '#F44336';
      default: return COLORS.textSecondary;
    }
  };

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: COLORS.background }}>
      <ScrollView contentContainerStyle={{ padding: 20 }}>
        {/* Header */}
        <Text style={{
          fontSize: 28,
          fontWeight: '700',
          color: COLORS.text,
          marginBottom: 24,
        }}>
          Profile
        </Text>

        {/* User Card */}
        <View style={{
          backgroundColor: COLORS.surface,
          borderRadius: 16,
          padding: 20,
          marginBottom: 24,
          borderWidth: 1,
          borderColor: COLORS.border,
        }}>
          <View style={{ flexDirection: 'row', alignItems: 'center' }}>
            {/* Avatar */}
            <View style={{
              width: 72,
              height: 72,
              borderRadius: 36,
              backgroundColor: COLORS.primaryDim,
              justifyContent: 'center',
              alignItems: 'center',
              marginRight: 16,
            }}>
              {user?.profilePhoto ? (
                <Image
                  source={{ uri: user.profilePhoto }}
                  style={{ width: 72, height: 72, borderRadius: 36 }}
                />
              ) : (
                <Text style={{ fontSize: 28, color: COLORS.primary, fontWeight: '600' }}>
                  {user?.name?.charAt(0) || 'U'}{user?.surname?.charAt(0) || ''}
                </Text>
              )}
            </View>
            
            {/* User Info */}
            <View style={{ flex: 1 }}>
              <Text style={{ color: COLORS.text, fontSize: 20, fontWeight: '600' }}>
                {user?.name} {user?.surname}
              </Text>
              <Text style={{ color: COLORS.textSecondary, fontSize: 14, marginTop: 2 }}>
                @{user?.username || 'user'}
              </Text>
              <Text style={{ color: COLORS.textSecondary, fontSize: 13, marginTop: 4 }}>
                {user?.email}
              </Text>
            </View>
          </View>

          {/* Stats Row */}
          <View style={{
            flexDirection: 'row',
            marginTop: 20,
            paddingTop: 16,
            borderTopWidth: 1,
            borderTopColor: COLORS.border,
          }}>
            <View style={{ flex: 1, alignItems: 'center' }}>
              <Text style={{ color: COLORS.textSecondary, fontSize: 12 }}>Status</Text>
              <View style={{
                backgroundColor: user?.status === 'ACTIVE' ? COLORS.primaryDim : COLORS.dangerDim,
                paddingHorizontal: 12,
                paddingVertical: 4,
                borderRadius: 12,
                marginTop: 4,
              }}>
                <Text style={{
                  color: user?.status === 'ACTIVE' ? COLORS.primary : COLORS.danger,
                  fontSize: 12,
                  fontWeight: '600',
                }}>
                  {user?.status || 'ACTIVE'}
                </Text>
              </View>
            </View>
            <View style={{ flex: 1, alignItems: 'center' }}>
              <Text style={{ color: COLORS.textSecondary, fontSize: 12 }}>Risk Profile</Text>
              <Text style={{
                color: getRiskProfileColor(user?.riskProfile),
                fontSize: 14,
                fontWeight: '600',
                marginTop: 4,
              }}>
                {user?.riskProfile || 'MODERATE'}
              </Text>
            </View>
            <View style={{ flex: 1, alignItems: 'center' }}>
              <Text style={{ color: COLORS.textSecondary, fontSize: 12 }}>Trading</Text>
              <Text style={{
                color: user?.tradingEnabled ? COLORS.primary : COLORS.danger,
                fontSize: 14,
                fontWeight: '600',
                marginTop: 4,
              }}>
                {user?.tradingEnabled ? 'Enabled' : 'Disabled'}
              </Text>
            </View>
          </View>
        </View>

        {/* Subscription Card */}
        {subscription && (
          <View style={{
            backgroundColor: COLORS.primaryDim,
            borderRadius: 12,
            padding: 16,
            marginBottom: 24,
            borderWidth: 1,
            borderColor: COLORS.primary,
          }}>
            <View style={{ flexDirection: 'row', alignItems: 'center' }}>
              <TrendingUp size={20} color={COLORS.primary} />
              <Text style={{ color: COLORS.primary, fontWeight: '600', marginLeft: 8 }}>
                {subscription.plan} Plan
              </Text>
            </View>
          </View>
        )}

        {/* Menu Items */}
        <Text style={{
          color: COLORS.textSecondary,
          fontSize: 13,
          fontWeight: '600',
          marginBottom: 12,
          textTransform: 'uppercase',
        }}>
          Account
        </Text>

        <MenuItem
          icon={User}
          title="Edit Profile"
          subtitle="Update your personal information"
          onPress={() => {}}
        />
        <MenuItem
          icon={Wallet}
          title="Linked Accounts"
          subtitle="Manage connected brokers"
          onPress={() => {}}
        />
        <MenuItem
          icon={Shield}
          title="Security"
          subtitle="Password, 2FA, login history"
          onPress={() => {}}
        />

        <Text style={{
          color: COLORS.textSecondary,
          fontSize: 13,
          fontWeight: '600',
          marginTop: 16,
          marginBottom: 12,
          textTransform: 'uppercase',
        }}>
          Preferences
        </Text>

        <MenuItem
          icon={Bell}
          title="Notifications"
          subtitle="Alerts, signals, market updates"
          onPress={() => {}}
        />
        <MenuItem
          icon={Settings}
          title="App Settings"
          subtitle="Theme, language, data"
          onPress={() => {}}
        />
        <MenuItem
          icon={HelpCircle}
          title="Help & Support"
          subtitle="FAQs, contact us"
          onPress={() => {}}
        />

        {/* Logout */}
        <View style={{ marginTop: 24 }}>
          <MenuItem
            icon={LogOut}
            title="Sign Out"
            onPress={handleLogout}
            danger
            showChevron={false}
          />
        </View>

        {/* Version */}
        <Text style={{
          color: COLORS.textSecondary,
          fontSize: 12,
          textAlign: 'center',
          marginTop: 24,
        }}>
          NeuralTrade v1.0.0
        </Text>
      </ScrollView>
    </SafeAreaView>
  );
}
