import { Module } from '@nestjs/common';
import { PortfolioService } from './portfolio.service';
import { PortfolioController } from './portfolio.controller';
import { PortfolioSnapshotService } from './snapshot.service';
import { AuthModule } from 'src/auth/auth.module';

@Module({
    imports: [
        AuthModule,
    ],
    controllers: [PortfolioController],
    providers: [PortfolioService, PortfolioSnapshotService],
    exports: [PortfolioService, PortfolioSnapshotService],
})
export class PortfolioModule { }

