export interface PaginatedResponse<T> {
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
