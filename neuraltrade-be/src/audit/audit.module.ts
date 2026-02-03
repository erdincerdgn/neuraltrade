import { Module } from '@nestjs/common';
import { AuditLogService } from './audit-log.service';
import { AuditLogController } from './audit-log.controller';
import { AuthModule } from 'src/auth/auth.module';

/**
 * Audit Module
 * 
 * Phase 3: Compliance & regulatory audit trail
 */
@Module({
    imports: [AuthModule],
    controllers: [AuditLogController],
    providers: [AuditLogService],
    exports: [AuditLogService],
})
export class AuditModule { }
