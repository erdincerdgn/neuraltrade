export interface RefreshTokenApiResponse {
  expiresIn: string;
  payload: {
    username: string;
    email: string;
    sub: number;
    preferedLanguageId: number;
    role: string;
  };
  authToken: string;
  refreshToken: string;
}

export interface RefreshTokenApiInput {
  refreshToken: string;
}

export interface LoginApiResponsePayload {
  username: string;
  email: string;
  sub: number;
  preferedLanguageId: number;
  role: string;
}

export interface LoginApiResponse {
  expiresIn: string;
  payload: LoginApiResponsePayload;
  authToken: string;
  refreshToken: string;
}
