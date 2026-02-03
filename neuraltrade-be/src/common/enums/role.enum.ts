// DEPRECATED: Use UserRole from '@prisma/client' instead
// This file is kept for backward compatibility only

// Re-export Prisma UserRole as the single source of truth
export { UserRole as Role } from '@prisma/client';

// Also export UserStatus and RiskProfile for convenience
export { UserRole, UserStatus, RiskProfile } from '@prisma/client';