/**
 * RBAC Permissions
 * 
 * Fine-grained permissions for NeuralTrade platform
 */
export enum Permission {
    // User Management
    USER_READ = 'user:read',
    USER_WRITE = 'user:write',
    USER_DELETE = 'user:delete',
    USER_MANAGE = 'user:manage',

    // Trading
    TRADE_READ = 'trade:read',
    TRADE_EXECUTE = 'trade:execute',
    TRADE_MANAGE = 'trade:manage',

    // Portfolio
    PORTFOLIO_READ = 'portfolio:read',
    PORTFOLIO_WRITE = 'portfolio:write',
    PORTFOLIO_MANAGE = 'portfolio:manage',

    // AI/Signals
    SIGNAL_READ = 'signal:read',
    SIGNAL_EXECUTE = 'signal:execute',
    MODEL_MANAGE = 'model:manage',

    // Risk Management
    RISK_READ = 'risk:read',
    RISK_CONFIGURE = 'risk:configure',

    // Admin
    ADMIN_READ = 'admin:read',
    ADMIN_WRITE = 'admin:write',
    ADMIN_FULL = 'admin:full',

    // Audit
    AUDIT_READ = 'audit:read',
    AUDIT_EXPORT = 'audit:export',

    // System
    SYSTEM_CONFIG = 'system:config',
    SYSTEM_MONITOR = 'system:monitor',
}

/**
 * User Roles
 */
export enum Role {
    USER = 'USER',
    TRADER = 'TRADER',
    ANALYST = 'ANALYST',
    MANAGER = 'MANAGER',
    ADMIN = 'ADMIN',
    SUPER_ADMIN = 'SUPER_ADMIN',
}

/**
 * Role Permission Mapping
 */
export const RolePermissions: Record<Role, Permission[]> = {
    [Role.USER]: [
        Permission.USER_READ,
        Permission.PORTFOLIO_READ,
        Permission.SIGNAL_READ,
    ],

    [Role.TRADER]: [
        Permission.USER_READ,
        Permission.TRADE_READ,
        Permission.TRADE_EXECUTE,
        Permission.PORTFOLIO_READ,
        Permission.PORTFOLIO_WRITE,
        Permission.SIGNAL_READ,
        Permission.SIGNAL_EXECUTE,
        Permission.RISK_READ,
    ],

    [Role.ANALYST]: [
        Permission.USER_READ,
        Permission.TRADE_READ,
        Permission.PORTFOLIO_READ,
        Permission.SIGNAL_READ,
        Permission.RISK_READ,
        Permission.AUDIT_READ,
    ],

    [Role.MANAGER]: [
        Permission.USER_READ,
        Permission.USER_WRITE,
        Permission.TRADE_READ,
        Permission.TRADE_EXECUTE,
        Permission.TRADE_MANAGE,
        Permission.PORTFOLIO_READ,
        Permission.PORTFOLIO_WRITE,
        Permission.PORTFOLIO_MANAGE,
        Permission.SIGNAL_READ,
        Permission.SIGNAL_EXECUTE,
        Permission.RISK_READ,
        Permission.RISK_CONFIGURE,
        Permission.AUDIT_READ,
    ],

    [Role.ADMIN]: [
        Permission.USER_READ,
        Permission.USER_WRITE,
        Permission.USER_MANAGE,
        Permission.TRADE_READ,
        Permission.TRADE_EXECUTE,
        Permission.TRADE_MANAGE,
        Permission.PORTFOLIO_READ,
        Permission.PORTFOLIO_WRITE,
        Permission.PORTFOLIO_MANAGE,
        Permission.SIGNAL_READ,
        Permission.SIGNAL_EXECUTE,
        Permission.MODEL_MANAGE,
        Permission.RISK_READ,
        Permission.RISK_CONFIGURE,
        Permission.ADMIN_READ,
        Permission.ADMIN_WRITE,
        Permission.AUDIT_READ,
        Permission.AUDIT_EXPORT,
        Permission.SYSTEM_MONITOR,
    ],

    [Role.SUPER_ADMIN]: Object.values(Permission),
};

/**
 * Check if role has permission
 */
export function hasPermission(role: Role, permission: Permission): boolean {
    return RolePermissions[role]?.includes(permission) ?? false;
}

/**
 * Check if role has any of the permissions
 */
export function hasAnyPermission(role: Role, permissions: Permission[]): boolean {
    return permissions.some((p) => hasPermission(role, p));
}

/**
 * Check if role has all permissions
 */
export function hasAllPermissions(role: Role, permissions: Permission[]): boolean {
    return permissions.every((p) => hasPermission(role, p));
}
