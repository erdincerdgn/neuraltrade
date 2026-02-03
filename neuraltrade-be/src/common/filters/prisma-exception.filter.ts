import { ArgumentsHost, Catch, ExceptionFilter, HttpStatus } from "@nestjs/common";
import { PrismaClientKnownRequestError } from '@prisma/client/runtime/library';
// import { Response } from "express";
import { FastifyReply } from "fastify";

@Catch(PrismaClientKnownRequestError)
export class PrismaExceptionFilter implements ExceptionFilter {
    catch(exception: PrismaClientKnownRequestError, host: ArgumentsHost) {
        const ctx = host.switchToHttp();
        // Express.js'ten Response nesnesi yerine FastifyReply kullanılıyor
        const response = ctx.getResponse<FastifyReply>();

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
                message = 'The provided value is invalid';
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
                message = 'An unexpected error occurred.';
        }

        // Express.js'te kullanılan response nesnesi yerine FastifyReply kullanılıyor .json yerine .send
        response.status(status).send({
            statusCode: status,
            message,
            error: exception.code,
            timestamp: new Date().toISOString(),
        });
    }
}