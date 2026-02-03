export interface BaseResponseInterface<T> {
  success: boolean;
  message?: string;
  data?: T;
  errorCode?: string;
}

export function okResponse<T>(
  data: T,
  message = 'Success',
): BaseResponseInterface<T> {
  return {
    success: true,
    message,
    data,
  };
}

export function failResponse<T = null>(
  message = 'Error',
  errorCode = 'UNKNOWN_ERROR',
): BaseResponseInterface<T> {
  return {
    success: false,
    message,
    errorCode,
  };
}
