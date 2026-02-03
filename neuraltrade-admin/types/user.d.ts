import { AuthRole } from '@/types/common';
// import { ICompany } from '@/types/company';

export interface IEditUser {
  username: string;
  name: string;
  // companyId: string;
}
export interface IUser {
  id: string;
  email: string;
  name?: string;
  surname?: string;
  username: string;
  password: string;
  phoneNumber: string;
  profilePhoto: string;
  role: AuthRole;
  createdAt: Date;
  updateAt: Date;
  preferedLanguageId?: string;
  // officeId: string;
  // company?: ICompany;
  // companyId?: string;
}

export interface ILoginUser {
  email: string;
  password: string;
}

export interface IRefreshToken {
  refreshToken: string;
}

interface FetchUserParams {
  page?: number;
  limit?: number;
  searchTerm?: string;
  sortBy?: 'id' | 'email' | 'name' | 'surname' | 'username' | "phoneNumber" | 'createdAt';
  sortDirection?: 'asc' | 'desc';
}
