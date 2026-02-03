import { createParamDecorator, ExecutionContext } from '@nestjs/common';

/**
 * Custom decorator to get the current authenticated user from the request.
 * Usage: @CurrentUser() user  or  @CurrentUser('id') userId
 */
export const CurrentUser = createParamDecorator(
    (data: string, ctx: ExecutionContext) => {
        const request = ctx.switchToHttp().getRequest();
        const user = request.user;

        if (!user) {
            return null;
        }

        return data ? user[data] : user;
    },
);
