import { persist } from 'zustand/middleware';
import { create } from 'zustand';

interface ConfigStore {
  direction: 'ltr' | 'rtl';
  setDirection: (direction: 'ltr' | 'rtl') => void;
}

export const useConfigStore = create<ConfigStore>()(
  persist(
    (set) => ({
      direction: 'ltr',
      colorScheme: 'light',
      setDirection: (direction: 'ltr' | 'rtl') => set({ direction }),
    }),
    { name: 'config-storage' }
  )
);
