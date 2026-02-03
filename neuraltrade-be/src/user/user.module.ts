import { Module } from '@nestjs/common';
import { UserController } from './user.controller';
import { UserService } from './user.service';
import { AuthModule } from '../auth/auth.module';
import { CommonModule } from '../common/common.module';

// Core modules are global, no need to import them explicitly
// PrismaModule, RedisModule, BullMQModule are @Global()

@Module({
  imports: [
    AuthModule,
    CommonModule,
  ],
  controllers: [UserController],
  providers: [UserService],
  exports: [UserService],
})
export class UserModule { }