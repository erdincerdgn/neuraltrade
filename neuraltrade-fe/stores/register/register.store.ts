import { create } from 'zustand';

interface UserData {
  name: string;
  surname: string;
//   companyName: string;
  profilePhotoUrl?: string;
}

interface RegisterStore {
  type: 'admin' | null;
  userData: UserData | null;
  setRegisterData: (type: 'admin', userData: UserData) => void;
  clearRegisterData: () => void;
  resetRegister: () => void;
}

const initialState = {
  type: null as 'admin' | null,
  userData: null as UserData | null,
};

type SetFunction = (fn: (state: RegisterStore) => Partial<RegisterStore>) => void;

export const createRegisterStore = (set: SetFunction) => ({
  ...initialState,

  setRegisterData: (type: 'admin', userData: UserData) =>
    set(() => ({ type, userData })),

  clearRegisterData: () =>
    set(() => ({ type: null, userData: null })),

  resetRegister: () =>
    set(() => ({ ...initialState })),
});

export const useRegisterStore = create<RegisterStore>((set) => createRegisterStore(set));
