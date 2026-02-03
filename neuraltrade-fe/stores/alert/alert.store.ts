import { create } from 'zustand';

interface AlertInterface {
  isVisible: boolean;
  title?: string;
  description?: string;
  color?: string;
}

interface AlertStore {
  alert: AlertInterface;
  setAlert: (alert: AlertInterface) => void;
  resetAlert: () => void;
}

const initialState = {
  alert: { isVisible: false },
};

type SetFunction = (fn: (state: AlertStore) => Partial<AlertStore>) => void;

export const createAlertStore = (set: SetFunction) => ({
  ...initialState,

  setAlert: (alert: AlertInterface) =>
    set(() => ({ alert })),

  resetAlert: () =>
    set(() => ({ ...initialState })),
});

export const useAlertStore = create<AlertStore>((set) => createAlertStore(set));