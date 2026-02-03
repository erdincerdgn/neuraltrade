export interface IApiResponse<T> {
  success: boolean;
  result: T;
  error?: {
    message: string;
  };
}

export interface PublicResponse {
  id: number;
  createdAt?: string;
  updatedAt?: string;
}

export interface ListMeta {
  total: number;
  lastPage: number;
  currentPage: number;
  perPage: number;
  prev: number | null;
  next: number | null;
}

export interface ListResponse<T> {
  data: T[];
  meta: ListMeta;
}

export interface ErrorResponse {
  statusCode: number;
  success: boolean;
  error: {
    message: string;
  };
}

export interface PaginationQuery {
  page: number;
  perPage: number;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
}
