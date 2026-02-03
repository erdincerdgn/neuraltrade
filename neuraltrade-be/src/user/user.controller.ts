import {
    Body,
    Controller,
    Get,
    Param,
    ParseIntPipe,
    Patch,
    Put,
    Query,
    UseGuards,
} from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse, ApiBearerAuth, ApiBody, ApiParam } from '@nestjs/swagger';
import { UserService } from './user.service';
import { SearchUserDto } from './dto/search-user.dto';
import { UpdateUserProfileDto } from './dto/update-profile.dto';
import { AdminUpdateUserDto, UserProfileStatsDto, UserResponseDto } from './dto/user.dto';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { AdminAuthGuard } from '../auth/guards/admin-auth.guard';
import { RolesGuard, Roles } from '../auth/guards/roles.guard';
import { CurrentUser } from '../common/decorators/roles.decorator';
import { PaginationDto } from '../common/pagination/dto/pagination.dto';
import { UserRole } from '@prisma/client';

@ApiTags('Users')
@Controller('user')
export class UserController {
    constructor(private readonly userService: UserService) { }

    // ==========================================
    // AUTHENTICATED USER ENDPOINTS
    // ==========================================

    @Get('me/stats')
    @UseGuards(JwtAuthGuard)
    @ApiBearerAuth()
    @ApiOperation({ summary: 'Get current user trading stats' })
    @ApiResponse({ status: 200, description: 'User trading stats', type: UserProfileStatsDto })
    async getMyStats(@CurrentUser('id') userId: number): Promise<UserProfileStatsDto> {
        return this.userService.getUserStats(userId);
    }

    @Patch('me/profile')
    @UseGuards(JwtAuthGuard)
    @ApiBearerAuth()
    @ApiOperation({ summary: 'Update current user profile' })
    @ApiResponse({ status: 200, description: 'Profile updated successfully' })
    @ApiResponse({ status: 409, description: 'Email/username already taken' })
    @ApiBody({ type: UpdateUserProfileDto })
    async updateMyProfile(
        @CurrentUser('id') userId: number,
        @Body() dto: UpdateUserProfileDto,
    ) {
        return this.userService.updateProfile(userId, dto);
    }

    // ==========================================
    // PUBLIC USER ENDPOINTS
    // ==========================================

    @Get(':username')
    @ApiOperation({ summary: 'Get public user profile by username' })
    @ApiParam({ name: 'username', description: 'Username' })
    @ApiResponse({ status: 200, description: 'User profile', type: UserResponseDto })
    @ApiResponse({ status: 404, description: 'User not found' })
    async getByUsername(@Param('username') username: string) {
        return this.userService.findByUsername(username);
    }

    // ==========================================
    // ADMIN ENDPOINTS
    // ==========================================

    @Get('admin/list')
    @UseGuards(AdminAuthGuard, RolesGuard)
    @Roles(UserRole.SUPER_ADMIN, UserRole.NEURALTRADE)
    @ApiBearerAuth()
    @ApiOperation({ summary: '[Admin] Get paginated user list' })
    @ApiResponse({ status: 200, description: 'Paginated user list' })
    async getUsers(@Query() paginationDto: PaginationDto) {
        return this.userService.findAll(paginationDto);
    }

    @Get('admin/search')
    @UseGuards(AdminAuthGuard, RolesGuard)
    @Roles(UserRole.SUPER_ADMIN, UserRole.NEURALTRADE)
    @ApiBearerAuth()
    @ApiOperation({ summary: '[Admin] Search users with filters' })
    @ApiResponse({ status: 200, description: 'Search results' })
    async searchUsers(@Query() searchDto: SearchUserDto) {
        return this.userService.searchUsers(searchDto);
    }

    @Get('admin/:id')
    @UseGuards(AdminAuthGuard, RolesGuard)
    @Roles(UserRole.SUPER_ADMIN, UserRole.NEURALTRADE)
    @ApiBearerAuth()
    @ApiOperation({ summary: '[Admin] Get user detail by ID' })
    @ApiParam({ name: 'id', description: 'User ID' })
    @ApiResponse({ status: 200, description: 'User detail' })
    @ApiResponse({ status: 404, description: 'User not found' })
    async getUserById(@Param('id', ParseIntPipe) id: number) {
        return this.userService.findOneById(id);
    }

    @Patch('admin/:id')
    @UseGuards(AdminAuthGuard, RolesGuard)
    @Roles(UserRole.SUPER_ADMIN, UserRole.NEURALTRADE)
    @ApiBearerAuth()
    @ApiOperation({ summary: '[Admin] Update user (status, role, trading)' })
    @ApiParam({ name: 'id', description: 'User ID' })
    @ApiBody({ type: AdminUpdateUserDto })
    @ApiResponse({ status: 200, description: 'User updated successfully' })
    async adminUpdateUser(
        @Param('id', ParseIntPipe) id: number,
        @Body() dto: AdminUpdateUserDto,
    ) {
        return this.userService.adminUpdateUser(id, dto);
    }

    @Put('admin/:id/trading/enable')
    @UseGuards(AdminAuthGuard, RolesGuard)
    @Roles(UserRole.SUPER_ADMIN, UserRole.NEURALTRADE)
    @ApiBearerAuth()
    @ApiOperation({ summary: '[Admin] Enable user trading' })
    @ApiParam({ name: 'id', description: 'User ID' })
    @ApiResponse({ status: 200, description: 'Trading enabled' })
    async enableTrading(@Param('id', ParseIntPipe) id: number) {
        return this.userService.toggleUserTrading(id, true);
    }

    @Put('admin/:id/trading/disable')
    @UseGuards(AdminAuthGuard, RolesGuard)
    @Roles(UserRole.SUPER_ADMIN, UserRole.NEURALTRADE)
    @ApiBearerAuth()
    @ApiOperation({ summary: '[Admin] Disable user trading (circuit breaker)' })
    @ApiParam({ name: 'id', description: 'User ID' })
    @ApiResponse({ status: 200, description: 'Trading disabled' })
    async disableTrading(@Param('id', ParseIntPipe) id: number) {
        return this.userService.toggleUserTrading(id, false);
    }
}