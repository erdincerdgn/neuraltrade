import { Injectable, CanActivate, ExecutionContext, ForbiddenException } from '@nestjs/common';
import { Reflector } from '@nestjs/core';
import { ROLES_KEY, PERMISSIONS_KEY, IS_PUBLIC_KEY } from './decorators';
import { Role, Permission, hasPermission, hasAnyPermission } from './permissions';

/**
 * Roles Guard
 * 
 * RBAC authorization guard that checks:
 * 1. User authentication
 * 2. Role-based access
 * 3. Permission-based access
 */
@Injectable()
export class RolesGuard implements CanActivate {
    constructor(private reflector: Reflector) { }

    canActivate(context: ExecutionContext): boolean {
        // Check if route is public
        const isPublic = this.reflector.getAllAndOverride<boolean>(IS_PUBLIC_KEY, [
            context.getHandler(),
            context.getClass(),
        ]);

        if (isPublic) {
            return true;
        }

        // Get required roles and permissions
        const requiredRoles = this.reflector.getAllAndOverride<Role[]>(ROLES_KEY, [
            context.getHandler(),
            context.getClass(),
        ]);

        const requiredPermissions = this.reflector.getAllAndOverride<Permission[]>(PERMISSIONS_KEY, [
            context.getHandler(),
            context.getClass(),
        ]);

        // If no roles or permissions required, allow access
        if (!requiredRoles?.length && !requiredPermissions?.length) {
            return true;
        }

        const request = context.switchToHttp().getRequest();
        const user = request.user;

        if (!user) {
            throw new ForbiddenException('User not authenticated');
        }

        // Check role-based access
        if (requiredRoles?.length) {
            const userRole = user.role as Role;
            const hasRole = requiredRoles.includes(userRole);

            if (!hasRole) {
                throw new ForbiddenException(
                    `Role ${userRole} does not have access. Required: ${requiredRoles.join(', ')}`,
                );
            }
        }

        // Check permission-based access
        if (requiredPermissions?.length) {
            const userRole = user.role as Role;
            const hasRequiredPermission = hasAnyPermission(userRole, requiredPermissions);

            if (!hasRequiredPermission) {
                throw new ForbiddenException(
                    `Insufficient permissions. Required: ${requiredPermissions.join(', ')}`,
                );
            }
        }

        return true;
    }
}

/**
 * Permission Check Utility
 * 
 * For programmatic permission checks in services
 */
@Injectable()
export class PermissionService {
    checkPermission(userRole: Role, permission: Permission): boolean {
        return hasPermission(userRole, permission);
    }

    checkPermissions(userRole: Role, permissions: Permission[]): boolean {
        return hasAnyPermission(userRole, permissions);
    }

    assertPermission(userRole: Role, permission: Permission): void {
        if (!this.checkPermission(userRole, permission)) {
            throw new ForbiddenException(`Permission denied: ${permission}`);
        }
    }
}
