// import { UserStatus } from "@prisma/client";

import { UserStatus } from "@prisma/client";

export interface PaginatedResult<T> {
  items: T[];
  meta: {
    total: number;
    page: number;
    lastPage: number;
    perPage: number;
    prev: number | null;
    next: number | null;
  };
}

export interface PaginationParams {
  page?: number;
  perPage?: number;
  search?: string;
  sortBy?: string;
  sortDirection?: 'asc' | 'desc';
  startDate?: string;
  endDate?: string;
  userStatus?: UserStatus; 
}
