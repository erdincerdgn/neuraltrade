import {
    Injectable,
    CanActivate,
    ExecutionContext,
    ForbiddenException,
    SetMetadata,
} from '@nestjs/common';
import { Reflector } from '@nestjs/core';
import { UserRole } from '@prisma/client';

/**
 * Roles decorator - Use with RolesGuard
 * 
 * Usage: @Roles(UserRole.SUPER_ADMIN, UserRole.NEURALTRADE)
 */
export const ROLES_KEY = 'roles';
export const Roles = (...roles: UserRole[]) => SetMetadata(ROLES_KEY, roles);

/**
 * Roles Guard - Role-Based Access Control (RBAC)
 * 
 * Use after JwtAuthGuard to check user roles.
 * 
 * Usage:
 * @UseGuards(JwtAuthGuard, RolesGuard)
 * @Roles(UserRole.SUPER_ADMIN)
 */
@Injectable()
export class RolesGuard implements CanActivate {
    constructor(private reflector: Reflector) { }

    canActivate(context: ExecutionContext): boolean {
        // Get required roles from decorator
        const requiredRoles = this.reflector.getAllAndOverride<UserRole[]>(ROLES_KEY, [
            context.getHandler(),
            context.getClass(),
        ]);

        // No roles required - allow access
        if (!requiredRoles || requiredRoles.length === 0) {
            return true;
        }

        // Get user from request
        const request = context.switchToHttp().getRequest();
        const user = request.user;

        if (!user) {
            throw new ForbiddenException('Authentication required');
        }

        // Check if user has required role
        const hasRole = requiredRoles.includes(user.role);

        if (!hasRole) {
            throw new ForbiddenException(
                `Access denied. Required roles: ${requiredRoles.join(', ')}. Your role: ${user.role}`
            );
        }

        return true;
    }
}

/**
 * Subscription Feature Guard
 * 
 * Check if user's subscription includes a specific feature
 */
export const FEATURE_KEY = 'requiredFeature';
export const RequiresFeature = (feature: string) => SetMetadata(FEATURE_KEY, feature);

@Injectable()
export class SubscriptionGuard implements CanActivate {
    constructor(private reflector: Reflector) { }

    async canActivate(context: ExecutionContext): Promise<boolean> {
        const requiredFeature = this.reflector.get<string>(FEATURE_KEY, context.getHandler());

        // No feature required - allow access
        if (!requiredFeature) {
            return true;
        }

        const request = context.switchToHttp().getRequest();
        const user = request.user;

        if (!user) {
            throw new ForbiddenException('Authentication required');
        }

        // Check subscription features
        // This should be loaded with user in JwtAuthGuard or fetched here
        const subscription = user.subscription;

        if (!subscription) {
            throw new ForbiddenException(
                `This feature requires a subscription. Please upgrade your plan.`
            );
        }

        // Map feature names to subscription fields
        const featureMap: Record<string, string> = {
            'ai_signals': 'aiSignalsEnabled',
            'quantum': 'quantumEnabled',
            'swarm': 'swarmEnabled',
            'rag': 'ragEnabled',
            'api_access': 'apiAccessEnabled',
        };

        const subscriptionField = featureMap[requiredFeature];

        if (subscriptionField && !subscription.plan?.[subscriptionField]) {
            throw new ForbiddenException(
                `The "${requiredFeature}" feature is not included in your current plan. ` +
                `Please upgrade to access this feature.`
            );
        }

        return true;
    }
}
