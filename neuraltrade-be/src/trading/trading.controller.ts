import {
    Controller,
    Get,
    Post,
    Delete,
    Body,
    Param,
    Query,
    UseGuards,
    HttpCode,
    HttpStatus,
    ParseIntPipe,
    Request,
} from '@nestjs/common';
import {
    ApiTags,
    ApiOperation,
    ApiBearerAuth,
    ApiParam,
    ApiQuery,
} from '@nestjs/swagger';
import { TradingService } from './trading.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { TradingGuard } from '../auth/guards/trading.guard';
import { CreateOrderDto } from './dto';

@ApiTags('Trading')
@Controller({ path: 'trading', version: '1' })
@UseGuards(JwtAuthGuard)
@ApiBearerAuth()
export class TradingController {
    constructor(private readonly tradingService: TradingService) { }

    // ==========================================
    // ORDERS
    // ==========================================

    @Post('orders/:portfolioId')
    @UseGuards(TradingGuard)
    @HttpCode(HttpStatus.CREATED)
    @ApiOperation({ summary: 'Place a new order' })
    @ApiParam({ name: 'portfolioId', description: 'Portfolio ID' })
    async placeOrder(
        @Request() req,
        @Param('portfolioId', ParseIntPipe) portfolioId: number,
        @Body() dto: CreateOrderDto,
    ) {
        return this.tradingService.placeOrder(req.user.id, portfolioId, dto);
    }

    @Delete('orders/:orderId')
    @ApiOperation({ summary: 'Cancel an order' })
    @ApiParam({ name: 'orderId', description: 'Order ID' })
    async cancelOrder(
        @Request() req,
        @Param('orderId', ParseIntPipe) orderId: number,
    ) {
        return this.tradingService.cancelOrder(req.user.id, orderId);
    }

    @Get('orders/:portfolioId/history')
    @ApiOperation({ summary: 'Get order history' })
    @ApiParam({ name: 'portfolioId', description: 'Portfolio ID' })
    @ApiQuery({ name: 'limit', required: false, type: Number })
    async getOrderHistory(
        @Request() req,
        @Param('portfolioId', ParseIntPipe) portfolioId: number,
        @Query('limit') limit?: number,
    ) {
        return this.tradingService.getOrderHistory(req.user.id, portfolioId, limit);
    }

    // ==========================================
    // POSITIONS
    // ==========================================

    @Get('positions/:portfolioId')
    @ApiOperation({ summary: 'Get open positions' })
    @ApiParam({ name: 'portfolioId', description: 'Portfolio ID' })
    async getPositions(
        @Request() req,
        @Param('portfolioId', ParseIntPipe) portfolioId: number,
    ) {
        return this.tradingService.getPositions(req.user.id, portfolioId);
    }

    @Get('positions/:portfolioId/summary')
    @ApiOperation({ summary: 'Get position summary' })
    @ApiParam({ name: 'portfolioId', description: 'Portfolio ID' })
    async getPositionSummary(@Param('portfolioId', ParseIntPipe) portfolioId: number) {
        return this.tradingService.getPositionSummary(portfolioId);
    }

    // ==========================================
    // STATISTICS & RISK
    // ==========================================

    @Get('stats/:portfolioId')
    @ApiOperation({ summary: 'Get trading statistics' })
    @ApiParam({ name: 'portfolioId', description: 'Portfolio ID' })
    async getTradingStats(
        @Request() req,
        @Param('portfolioId', ParseIntPipe) portfolioId: number,
    ) {
        return this.tradingService.getTradingStats(req.user.id, portfolioId);
    }

    @Get('risk/:portfolioId')
    @ApiOperation({ summary: 'Get portfolio risk metrics' })
    @ApiParam({ name: 'portfolioId', description: 'Portfolio ID' })
    async getRiskMetrics(@Param('portfolioId', ParseIntPipe) portfolioId: number) {
        return this.tradingService.getRiskMetrics(portfolioId);
    }

    @Get('trades/:portfolioId/history')
    @ApiOperation({ summary: 'Get trade history' })
    @ApiParam({ name: 'portfolioId', description: 'Portfolio ID' })
    @ApiQuery({ name: 'limit', required: false, type: Number })
    async getTradeHistory(
        @Request() req,
        @Param('portfolioId', ParseIntPipe) portfolioId: number,
        @Query('limit') limit?: number,
    ) {
        return this.tradingService.getTradeHistory(req.user.id, portfolioId, limit);
    }
}
