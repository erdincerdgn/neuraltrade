// Re-export from auth/guards to avoid duplication
// The main decorators are maintained in the auth module
export { Roles, ROLES_KEY, RequiresFeature } from '../../auth/guards/roles.guard';

// Additional common decorators
import { createParamDecorator, ExecutionContext, SetMetadata } from '@nestjs/common';

/**
 * CurrentUser decorator - Get current authenticated user from request
 * Usage: @CurrentUser() user: User or @CurrentUser('id') userId: number
 */
export const CurrentUser = createParamDecorator(
    (data: string | undefined, ctx: ExecutionContext) => {
        const request = ctx.switchToHttp().getRequest();
        const user = request.user;

        if (!user) {
            return null;
        }

        return data ? user[data] : user;
    },
);

/**
 * Public decorator - Mark endpoint as public (skip auth)
 */
export const IS_PUBLIC_KEY = 'isPublic';
export const Public = () => SetMetadata(IS_PUBLIC_KEY, true);

/**
 * ApiRequest decorator - Get full request object
 */
export const ApiRequest = createParamDecorator(
    (_data: unknown, ctx: ExecutionContext) => {
        return ctx.switchToHttp().getRequest();
    },
);