import { ArgumentsHost, Catch, ExceptionFilter, HttpException, HttpStatus } from "@nestjs/common";
import { PrismaClientKnownRequestError, PrismaClientValidationError } from '@prisma/client/runtime/library';
import { FastifyReply } from 'fastify';

@Catch()
export class AllExceptionsFilter implements ExceptionFilter {
    catch(exception: unknown, host: ArgumentsHost) {
        const ctx = host.switchToHttp();
        const response = ctx.getResponse<FastifyReply>();

        let status = HttpStatus.INTERNAL_SERVER_ERROR;
        let message = 'Internal server error';
        let error = null;

        // Prisma error handling
        if (exception instanceof PrismaClientKnownRequestError) {
            // Handle known Prisma errors
            const prismaError = this.handlePrismaError(exception);
            status = prismaError.status;
            message = prismaError.message;
            error = prismaError.error;
        }

        // NestJS HTTP Exceptions
        else if (exception instanceof HttpException) {
            status = exception.getStatus();
            const exceptionResponse = exception.getResponse();
            message = 
            typeof exceptionResponse === 'string'
                ? exceptionResponse 
                : (exceptionResponse as any).message || exception.message;
        }

        // Prisma Validation Error
        else if (exception instanceof PrismaClientValidationError) {
            status = HttpStatus.BAD_REQUEST;
            const validationError = this.handlePrismaValidationError(
                exception.message,
            );
            message = validationError.message;
            error = validationError.error;
        }

        // Unknown Errors
        else if (exception instanceof Error) {
            message = exception.message;
            error = exception.stack;
        }

        // Log the error for debugging
        console.error('Exception:', {
            status,
            message,
            error,
            exception,
        });

        response.status(status).send({
            statusCode: status,
            message,
            error: error || undefined,
            timestamp: new Date().toISOString(),
        });
    }

    private handlePrismaValidationError(errorMessage: string): {
        message: string;
        error: string;
    } {
        // Error handling for enum value
        if (errorMessage.includes('Invalid value for argument')) {
            const fieldMatch = errorMessage.match(
                /Invalid value for argument `([^`]+)`/,
            );
            
            const enumMatch = errorMessage.match(/Expected ([A-Za-z]+)\./);

            if (fieldMatch && enumMatch) {
                const fieldName = fieldMatch[1];
                const enumType = enumMatch[1];
                return {
                    message: `Invalid value for field '${fieldName}'. Expected a valid '${enumType}' enum value.`,
                    error: `INVALID_ENUM_VALUE_${enumType.toUpperCase()}`,
                };
            }
        }

        // Other validation errors
        return {
            message: 'Data validation error',
            error: 'VALIDATION_ERROR',
        };
    }


    private handlePrismaError(exception: PrismaClientKnownRequestError): {
        status: HttpStatus;
        message: string;
        error: string;
    } {
        let status = HttpStatus.BAD_REQUEST;
        let message = 'Validation error';

        switch (exception.code) {
            case 'P2000':
                message = 'The provided value for the column is too long for the column\'s type.';
                break;
            case 'P2002':
                message = 'The unique restriction for this area will be interrupted';
                break;
            case 'P2003':
                message = 'Foreign key constraint failed';
                break;
            case 'P2005':
                message = 'Invalid value for field type';
                break;
            case 'P2006':
                message = 'The value provided is invalid';
                if (exception.message.includes('Enum')) {
                    const enumType = exception.message.match(/`([^`]+)`/)?.[1] || 'field';
                    message = `Invalid value for field '${enumType}'. Expected a valid enum value.`;
                }
                break;
            case 'P2011':
                message = 'Field restriction that cannot be left blank';
                break;
            case 'P2012':
                message = 'A required field is missing';
                break;
            case 'P2025':
                status = HttpStatus.NOT_FOUND;
                message = 'No record found';
                break;
            default:
                status = HttpStatus.INTERNAL_SERVER_ERROR;
                message = 'An unexpected database error occurred.';
        }

        return {
            status,
            message,
            error: exception.code,
        };
    }
}