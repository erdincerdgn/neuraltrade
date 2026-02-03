import { create } from 'zustand';

interface ModalStore {
  isVisibleLogin: boolean;
  isVisibleSignup: boolean;
  setIsVisibleLogin: (visible: boolean) => void;
  setIsVisibleSignup: (visible: boolean) => void;
  resetModals: () => void;
}

const initialState = {
  isVisibleLogin: false,
  isVisibleSignup: false,
};

type SetFunction = (fn: (state: ModalStore) => Partial<ModalStore>) => void;

export const createModalStore = (set: SetFunction) => ({
  ...initialState,

  setIsVisibleLogin: (visible: boolean) =>
    set(() => ({ isVisibleLogin: visible })),

  setIsVisibleSignup: (visible: boolean) =>
    set(() => ({ isVisibleSignup: visible })),

  resetModals: () =>
    set(() => ({ ...initialState })),
});

export const useModal = create<ModalStore>((set) => createModalStore(set));
