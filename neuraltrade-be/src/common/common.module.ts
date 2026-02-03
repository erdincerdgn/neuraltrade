import { Global, Module } from '@nestjs/common';
import { APP_FILTER, APP_INTERCEPTOR } from '@nestjs/core';
import { PrismaModule } from '../core/prisma/prisma.module';

// Services
import { PaginationService } from './pagination/services/pagination.service';

// Filters
import { AllExceptionsFilter } from './filters/all-exceptions.filter';
import { PrismaExceptionFilter } from './filters/prisma-exception.filter';

// Interceptors
import { LoggingInterceptor } from './interceptors/logging.interceptor';

@Global()
@Module({
  imports: [PrismaModule],
  providers: [
    // Services
    PaginationService,

    // Global Exception Filters
    {
      provide: APP_FILTER,
      useClass: AllExceptionsFilter,
    },
    {
      provide: APP_FILTER,
      useClass: PrismaExceptionFilter,
    },

    // Global Interceptors
    {
      provide: APP_INTERCEPTOR,
      useClass: LoggingInterceptor,
    },
  ],
  exports: [PaginationService],
})
export class CommonModule { }
