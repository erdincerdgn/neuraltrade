import { ListResponse, IApiResponse } from '@/types/api';
import { RefreshTokenApiInput, RefreshTokenApiResponse } from '@/types/token';
import { IEditUser, IUser, FetchUserParams } from '@/types/user';
import { axiosWithoutToken, axiosWithToken } from '../axios';
import { LoginResponse } from '@/types/auth';
import { PaginatedResponse } from '@/types/pagination';

export const login = async (email: string, password: string) => {
  // Updated backend endpoint for admin login
  const response = await axiosWithoutToken.post<LoginResponse>('/auth/login-admin', {
    email,
    password,
  });
  return response.data;
};

export const refreshTokenApi = async (
  values: RefreshTokenApiInput
): Promise<RefreshTokenApiResponse> => {
  const response = await axiosWithToken.post<RefreshTokenApiResponse>('/auth/refresh', values);
  return response.data;
};

export async function fetchUserApi(params: FetchUserParams): Promise<PaginatedResponse<IUser>> {
  try {

    const query = new URLSearchParams();
    if (params.page) query.append('page', params.page.toString());
    if (params.limit) query.append('limit', params.limit.toString());
    if (params.searchTerm) query.append('searchTerm', params.searchTerm);
    if (params.sortBy) query.append('sortBy', params.sortBy);
    if (params.sortDirection) query.append('sortDirection', params.sortDirection);
    const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/user/search?${query.toString()}`);

    const json = await response.json();


    if (!response.ok) {
      console.error('API error:', json);
      throw new Error(json.message || 'Failed to fetch users');
    }
    return {
      items: json.data,
      meta: {
        total: json.total,
        page: json.page,
        lastPage: json.totalPages,
        perPage: json.limit,
        prev: json.page > 1 ? json.page - 1 : null,
        next: json.page < json.totalPages ? json.page + 1 : null,
      },
    };
  } catch (err) {
    console.error('Fetch error:', err);
    return {
      items: [],
      meta: {
        total: 0,
        page: 1,
        lastPage: 1,
        perPage: 10,
        prev: null,
        next: null,
      },
    };
  }
}

export const getUserDetails = async (userId: string): Promise<IUser> => {
  const response = await axiosWithToken.get(`/user/detail?id=${userId}`);
  return response.data;
}

export const editUserApi = async (
  id: string,
  values: IEditUser
): Promise<IApiResponse<IEditUser>> => {
  const response = await axiosWithToken.patch<IApiResponse<IEditUser>>(`/user/${id}`, values);
  return response.data;
};

export const deleteUserApi = async (id: string): Promise<IApiResponse<null>> => {
  const response = await axiosWithToken.delete<IApiResponse<null>>(`/user/${id}`);
  return response.data;
};
