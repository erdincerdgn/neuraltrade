/**
 * Register Screen
 * ================
 * New user registration form
 * Matches API schema: email, username, name, surname, password, phoneNumber, gender, dateOfBirth, riskProfile
 */

import React, { useState, useCallback, memo } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  Alert,
  TextInputProps,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import {
  Zap,
  Mail,
  Lock,
  User,
  Phone,
  Eye,
  EyeOff,
  ChevronDown,
  ArrowLeft,
  LucideIcon,
} from 'lucide-react-native';
import authService, { RegisterRequest } from '@/services/auth';
import { useAuthStore } from '@/store/auth.store';

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
  error: '#FF3B30',
};

// ============================================
// CONSTANTS
// ============================================
const RISK_PROFILES = [
  { value: 'LOW', label: 'Low Risk - Conservative' },
  { value: 'MODERATE', label: 'Moderate Risk - Balanced' },
  { value: 'HIGH', label: 'High Risk - Growth' },
  { value: 'AGGRESSIVE', label: 'Aggressive - Maximum Growth' },
];

// ============================================
// INPUT FIELD COMPONENT (OUTSIDE OF MAIN COMPONENT)
// ============================================
interface InputFieldProps extends Omit<TextInputProps, 'style'> {
  icon: LucideIcon;
  error?: string;
  showPasswordToggle?: boolean;
  showPassword?: boolean;
  onTogglePassword?: () => void;
}

const InputField = memo(function InputField({
  icon: Icon,
  error,
  showPasswordToggle,
  showPassword,
  onTogglePassword,
  ...textInputProps
}: InputFieldProps) {
  return (
    <View style={{ marginBottom: 16 }}>
      <View style={{
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: COLORS.surface,
        borderRadius: 12,
        borderWidth: 1,
        borderColor: error ? COLORS.error : COLORS.border,
        paddingHorizontal: 16,
      }}>
        <Icon size={20} color={error ? COLORS.error : COLORS.textSecondary} />
        <TextInput
          style={{
            flex: 1,
            paddingVertical: 16,
            paddingHorizontal: 12,
            fontSize: 16,
            color: COLORS.text,
          }}
          placeholderTextColor={COLORS.textSecondary}
          autoCorrect={false}
          {...textInputProps}
        />
        {showPasswordToggle && onTogglePassword && (
          <TouchableOpacity onPress={onTogglePassword}>
            {showPassword ? (
              <EyeOff size={20} color={COLORS.textSecondary} />
            ) : (
              <Eye size={20} color={COLORS.textSecondary} />
            )}
          </TouchableOpacity>
        )}
      </View>
      {error && (
        <Text style={{ color: COLORS.error, fontSize: 12, marginTop: 4, marginLeft: 4 }}>
          {error}
        </Text>
      )}
    </View>
  );
});

// ============================================
// REGISTER SCREEN
// ============================================
export default function RegisterScreen() {
  const router = useRouter();
  const { setLoading, isLoading } = useAuthStore();

  // Form state - individual states instead of object to prevent re-renders
  const [step, setStep] = useState(1);
  const [email, setEmail] = useState('');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [name, setName] = useState('');
  const [surname, setSurname] = useState('');
  const [phoneNumber, setPhoneNumber] = useState('');
  const [riskProfile, setRiskProfile] = useState('MODERATE');
  
  const [showPassword, setShowPassword] = useState(false);
  const [showRiskPicker, setShowRiskPicker] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [agreedToTerms, setAgreedToTerms] = useState(false);

  // ============================================
  // VALIDATION
  // ============================================

  const validateStep1 = (): boolean => {
    const newErrors: Record<string, string> = {};

    if (!email.trim()) {
      newErrors.email = 'Email is required';
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      newErrors.email = 'Invalid email format';
    }

    if (!username.trim()) {
      newErrors.username = 'Username is required';
    } else if (username.length < 3) {
      newErrors.username = 'Username must be at least 3 characters';
    }

    if (!password) {
      newErrors.password = 'Password is required';
    } else if (password.length < 8) {
      newErrors.password = 'Password must be at least 8 characters';
    }

    if (password !== confirmPassword) {
      newErrors.confirmPassword = 'Passwords do not match';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const validateStep2 = (): boolean => {
    const newErrors: Record<string, string> = {};

    if (!name.trim()) {
      newErrors.name = 'First name is required';
    }

    if (!surname.trim()) {
      newErrors.surname = 'Last name is required';
    }

    if (!agreedToTerms) {
      newErrors.terms = 'You must agree to the terms';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  // ============================================
  // HANDLERS
  // ============================================

  const handleNextStep = useCallback(() => {
    if (validateStep1()) {
      setStep(2);
    }
  }, [email, username, password, confirmPassword]);

  const handleRegister = useCallback(async () => {
    if (!validateStep2()) return;

    setLoading(true);
    setErrors({});

    try {
      await authService.register({
        email: email.trim().toLowerCase(),
        username: username.trim(),
        name: name.trim(),
        surname: surname.trim(),
        password: password,
        phoneNumber: phoneNumber.trim() || undefined,
        gender: 'Unspecified',
        riskProfile: riskProfile as 'LOW' | 'MODERATE' | 'HIGH' | 'AGGRESSIVE',
      });

      Alert.alert(
        'Account Created! 🎉',
        'Welcome to NeuralTrade. Your AI trading journey begins now.',
        [{ text: 'Get Started', onPress: () => router.replace('/' as any) }]
      );
    } catch (err: any) {
      const message = err.response?.data?.message || 'Registration failed. Please try again.';
      setErrors({ submit: message });
      Alert.alert('Registration Failed', message);
    } finally {
      setLoading(false);
    }
  }, [email, username, name, surname, password, phoneNumber, riskProfile, router, setLoading, agreedToTerms]);

  const togglePassword = useCallback(() => {
    setShowPassword(prev => !prev);
  }, []);

  const clearError = useCallback((key: string) => {
    setErrors(prev => {
      if (!prev[key]) return prev;
      const next = { ...prev };
      delete next[key];
      return next;
    });
  }, []);

  // ============================================
  // RENDER STEP 1: ACCOUNT INFO
  // ============================================

  const renderStep1 = () => (
    <>
      <Text style={{
        fontSize: 24,
        fontWeight: '600',
        color: COLORS.text,
        marginBottom: 8,
      }}>
        Create Account
      </Text>
      <Text style={{
        fontSize: 16,
        color: COLORS.textSecondary,
        marginBottom: 32,
      }}>
        Step 1 of 2 - Account Information
      </Text>

      <InputField
        icon={Mail}
        placeholder="Email"
        value={email}
        onChangeText={(text) => { setEmail(text); clearError('email'); }}
        error={errors.email}
        keyboardType="email-address"
        autoCapitalize="none"
        editable={!isLoading}
      />

      <InputField
        icon={User}
        placeholder="Username"
        value={username}
        onChangeText={(text) => { setUsername(text); clearError('username'); }}
        error={errors.username}
        autoCapitalize="none"
        editable={!isLoading}
      />

      <InputField
        icon={Lock}
        placeholder="Password"
        value={password}
        onChangeText={(text) => { setPassword(text); clearError('password'); }}
        error={errors.password}
        secureTextEntry={!showPassword}
        showPasswordToggle
        showPassword={showPassword}
        onTogglePassword={togglePassword}
        editable={!isLoading}
      />

      <InputField
        icon={Lock}
        placeholder="Confirm Password"
        value={confirmPassword}
        onChangeText={(text) => { setConfirmPassword(text); clearError('confirmPassword'); }}
        error={errors.confirmPassword}
        secureTextEntry={!showPassword}
        editable={!isLoading}
      />

      <TouchableOpacity
        style={{
          backgroundColor: COLORS.primary,
          paddingVertical: 16,
          borderRadius: 12,
          alignItems: 'center',
          marginTop: 8,
        }}
        onPress={handleNextStep}
      >
        <Text style={{ color: '#000', fontSize: 18, fontWeight: '600' }}>
          Continue
        </Text>
      </TouchableOpacity>
    </>
  );

  // ============================================
  // RENDER STEP 2: PERSONAL INFO
  // ============================================

  const renderStep2 = () => (
    <>
      {/* Back Button */}
      <TouchableOpacity
        style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 24 }}
        onPress={() => setStep(1)}
      >
        <ArrowLeft size={20} color={COLORS.primary} />
        <Text style={{ color: COLORS.primary, marginLeft: 8 }}>Back</Text>
      </TouchableOpacity>

      <Text style={{
        fontSize: 24,
        fontWeight: '600',
        color: COLORS.text,
        marginBottom: 8,
      }}>
        Personal Details
      </Text>
      <Text style={{
        fontSize: 16,
        color: COLORS.textSecondary,
        marginBottom: 32,
      }}>
        Step 2 of 2 - Complete your profile
      </Text>

      <InputField
        icon={User}
        placeholder="First Name"
        value={name}
        onChangeText={(text) => { setName(text); clearError('name'); }}
        error={errors.name}
        autoCapitalize="words"
        editable={!isLoading}
      />

      <InputField
        icon={User}
        placeholder="Last Name"
        value={surname}
        onChangeText={(text) => { setSurname(text); clearError('surname'); }}
        error={errors.surname}
        autoCapitalize="words"
        editable={!isLoading}
      />

      <InputField
        icon={Phone}
        placeholder="Phone Number (optional)"
        value={phoneNumber}
        onChangeText={setPhoneNumber}
        keyboardType="phone-pad"
        editable={!isLoading}
      />

      {/* Risk Profile Picker */}
      <TouchableOpacity
        style={{
          flexDirection: 'row',
          alignItems: 'center',
          justifyContent: 'space-between',
          backgroundColor: COLORS.surface,
          borderRadius: 12,
          borderWidth: 1,
          borderColor: COLORS.border,
          paddingHorizontal: 16,
          paddingVertical: 16,
          marginBottom: 16,
        }}
        onPress={() => setShowRiskPicker(!showRiskPicker)}
      >
        <Text style={{ color: COLORS.text, fontSize: 16 }}>
          {RISK_PROFILES.find(r => r.value === riskProfile)?.label || 'Select Risk Profile'}
        </Text>
        <ChevronDown size={20} color={COLORS.textSecondary} />
      </TouchableOpacity>

      {showRiskPicker && (
        <View style={{
          backgroundColor: COLORS.surface,
          borderRadius: 12,
          marginBottom: 16,
          overflow: 'hidden',
        }}>
          {RISK_PROFILES.map(risk => (
            <TouchableOpacity
              key={risk.value}
              style={{
                paddingVertical: 14,
                paddingHorizontal: 16,
                backgroundColor: riskProfile === risk.value ? COLORS.primaryDim : 'transparent',
              }}
              onPress={() => {
                setRiskProfile(risk.value);
                setShowRiskPicker(false);
              }}
            >
              <Text style={{
                color: riskProfile === risk.value ? COLORS.primary : COLORS.text,
              }}>
                {risk.label}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      )}

      {/* Terms & Conditions */}
      <TouchableOpacity
        style={{
          flexDirection: 'row',
          alignItems: 'flex-start',
          marginBottom: 24,
        }}
        onPress={() => setAgreedToTerms(!agreedToTerms)}
      >
        <View style={{
          width: 20,
          height: 20,
          borderRadius: 4,
          borderWidth: 2,
          borderColor: errors.terms ? COLORS.error : (agreedToTerms ? COLORS.primary : COLORS.border),
          backgroundColor: agreedToTerms ? COLORS.primary : 'transparent',
          justifyContent: 'center',
          alignItems: 'center',
          marginRight: 12,
          marginTop: 2,
        }}>
          {agreedToTerms && <Text style={{ color: '#000', fontSize: 12, fontWeight: 'bold' }}>✓</Text>}
        </View>
        <Text style={{ flex: 1, color: COLORS.textSecondary, lineHeight: 20 }}>
          I agree to the{' '}
          <Text style={{ color: COLORS.primary }}>Terms of Service</Text>
          {' '}and{' '}
          <Text style={{ color: COLORS.primary }}>Privacy Policy</Text>
        </Text>
      </TouchableOpacity>
      {errors.terms && (
        <Text style={{ color: COLORS.error, fontSize: 12, marginBottom: 16, marginLeft: 4 }}>
          {errors.terms}
        </Text>
      )}

      {/* Submit Error */}
      {errors.submit && (
        <View style={{
          backgroundColor: 'rgba(255, 59, 48, 0.1)',
          padding: 12,
          borderRadius: 8,
          marginBottom: 16,
        }}>
          <Text style={{ color: COLORS.error, textAlign: 'center' }}>{errors.submit}</Text>
        </View>
      )}

      {/* Register Button */}
      <TouchableOpacity
        style={{
          backgroundColor: COLORS.primary,
          paddingVertical: 16,
          borderRadius: 12,
          alignItems: 'center',
          opacity: isLoading ? 0.6 : 1,
        }}
        onPress={handleRegister}
        disabled={isLoading}
      >
        {isLoading ? (
          <ActivityIndicator color="#000" />
        ) : (
          <Text style={{ color: '#000', fontSize: 18, fontWeight: '600' }}>
            Create Account
          </Text>
        )}
      </TouchableOpacity>
    </>
  );

  // ============================================
  // MAIN RENDER
  // ============================================

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: COLORS.background }}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={{ flex: 1 }}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 0 : 20}
      >
        <ScrollView
          contentContainerStyle={{ flexGrow: 1, padding: 24 }}
          keyboardShouldPersistTaps="handled"
          showsVerticalScrollIndicator={false}
        >
          {/* Logo */}
          <View style={{ alignItems: 'center', marginBottom: 32, marginTop: 20 }}>
            <View style={{
              width: 64,
              height: 64,
              borderRadius: 16,
              backgroundColor: COLORS.primaryDim,
              justifyContent: 'center',
              alignItems: 'center',
            }}>
              <Zap size={32} color={COLORS.primary} />
            </View>
          </View>

          {/* Form Steps */}
          {step === 1 ? renderStep1() : renderStep2()}

          {/* Login Link */}
          <View style={{
            flexDirection: 'row',
            justifyContent: 'center',
            marginTop: 32,
          }}>
            <Text style={{ color: COLORS.textSecondary }}>Already have an account? </Text>
            <TouchableOpacity onPress={() => router.push('/login' as any)}>
              <Text style={{ color: COLORS.primary, fontWeight: '600' }}>Sign In</Text>
            </TouchableOpacity>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}
