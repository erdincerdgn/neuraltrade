import {
    Controller,
    Get,
    Post,
    Patch,
    Delete,
    Body,
    Param,
    UseGuards,
    HttpCode,
    HttpStatus,
    ParseIntPipe,
    Request,
} from '@nestjs/common';
import {
    ApiTags,
    ApiOperation,
    ApiResponse,
    ApiBearerAuth,
    ApiParam,
} from '@nestjs/swagger';
import { PortfolioService } from './portfolio.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { CreatePortfolioDto, UpdatePortfolioDto } from './dto';

@ApiTags('Portfolio')
@Controller({ path: 'portfolio', version: '1' })
@UseGuards(JwtAuthGuard)
@ApiBearerAuth()
export class PortfolioController {
    constructor(private readonly portfolioService: PortfolioService) { }

    // ==========================================
    // PORTFOLIO CRUD
    // ==========================================

    @Post()
    @HttpCode(HttpStatus.CREATED)
    @ApiOperation({ summary: 'Create a new portfolio' })
    @ApiResponse({ status: 201, description: 'Portfolio created' })
    async createPortfolio(@Request() req, @Body() dto: CreatePortfolioDto) {
        return this.portfolioService.createPortfolio(req.user.id, dto);
    }

    @Get()
    @ApiOperation({ summary: 'Get all portfolios' })
    async getPortfolios(@Request() req) {
        return this.portfolioService.getPortfolios(req.user.id);
    }

    @Get(':id')
    @ApiOperation({ summary: 'Get portfolio by ID' })
    @ApiParam({ name: 'id', description: 'Portfolio ID' })
    async getPortfolio(@Request() req, @Param('id', ParseIntPipe) portfolioId: number) {
        return this.portfolioService.getPortfolioById(req.user.id, portfolioId);
    }

    @Patch(':id')
    @ApiOperation({ summary: 'Update portfolio' })
    @ApiParam({ name: 'id', description: 'Portfolio ID' })
    async updatePortfolio(
        @Request() req,
        @Param('id', ParseIntPipe) portfolioId: number,
        @Body() dto: UpdatePortfolioDto,
    ) {
        return this.portfolioService.updatePortfolio(req.user.id, portfolioId, dto);
    }

    @Delete(':id')
    @ApiOperation({ summary: 'Delete portfolio' })
    @ApiParam({ name: 'id', description: 'Portfolio ID' })
    async deletePortfolio(@Request() req, @Param('id', ParseIntPipe) portfolioId: number) {
        return this.portfolioService.deletePortfolio(req.user.id, portfolioId);
    }

    // ==========================================
    // PORTFOLIO OVERVIEW & ANALYTICS
    // ==========================================

    @Get(':id/overview')
    @ApiOperation({ summary: 'Get portfolio overview' })
    @ApiParam({ name: 'id', description: 'Portfolio ID' })
    async getPortfolioOverview(@Request() req, @Param('id', ParseIntPipe) portfolioId: number) {
        return this.portfolioService.getPortfolioOverview(req.user.id, portfolioId);
    }

    // ==========================================
    // HOLDINGS & P&L
    // ==========================================

    @Get(':id/holdings')
    @ApiOperation({ summary: 'Get portfolio holdings' })
    @ApiParam({ name: 'id', description: 'Portfolio ID' })
    async getHoldings(@Param('id', ParseIntPipe) portfolioId: number) {
        return this.portfolioService.getHoldings(portfolioId);
    }

    @Get(':id/allocation')
    @ApiOperation({ summary: 'Get portfolio allocation breakdown' })
    @ApiParam({ name: 'id', description: 'Portfolio ID' })
    async getAllocation(@Param('id', ParseIntPipe) portfolioId: number) {
        return this.portfolioService.getAllocation(portfolioId);
    }

    @Get(':id/pnl')
    @ApiOperation({ summary: 'Get P&L summary' })
    @ApiParam({ name: 'id', description: 'Portfolio ID' })
    async getPnLSummary(@Param('id', ParseIntPipe) portfolioId: number) {
        return this.portfolioService.getPnLSummary(portfolioId);
    }

    @Get(':id/risk')
    @ApiOperation({ summary: 'Get risk metrics' })
    @ApiParam({ name: 'id', description: 'Portfolio ID' })
    async getRiskMetrics(@Param('id', ParseIntPipe) portfolioId: number) {
        return this.portfolioService.getRiskMetrics(portfolioId);
    }
}
