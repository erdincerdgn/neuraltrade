export enum AuthStatus {
  Authenticated = 'authenticated',
  Unauthenticated = 'unauthenticated',
  Loading = 'loading',
}

export enum AuthRole {
  user = 'USER',
  neuraltrade = 'NEURALTRADE',
  superAdmin = 'SUPER_ADMIN',
  // companyAdmin = 'COMPANY_ADMIN',
}