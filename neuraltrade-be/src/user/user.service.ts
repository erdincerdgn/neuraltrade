import {
    Injectable,
    NotFoundException,
    ConflictException,
    Logger,
} from '@nestjs/common';
import { PrismaService } from '../core/prisma/prisma.service';
import { RedisService } from '../core/redis/redis.service';
import { QueueService } from '../core/bullmq/queue.service';
import { Prisma } from '@prisma/client';
import { PaginationService } from '../common/pagination/services/pagination.service';
import { PaginationParams } from '../common/pagination/interfaces/pagination.interface';
import { SearchUserDto } from './dto/search-user.dto';
import { UpdateUserProfileDto } from './dto/update-profile.dto';
import { UserProfileStatsDto, AdminUpdateUserDto } from './dto/user.dto';

@Injectable()
export class UserService {
    private readonly logger = new Logger(UserService.name);

    // Cache keys
    private readonly CACHE_KEYS = {
        USER_PROFILE: (id: number) => `user:profile:${id}`,
        USER_STATS: (id: number) => `user:stats:${id}`,
        USER_BY_USERNAME: (username: string) => `user:username:${username}`,
    };

    constructor(
        private prisma: PrismaService,
        private paginationService: PaginationService,
        private redis: RedisService,
        private queue: QueueService,
    ) { }

    // ==========================================
    // USER CRUD
    // ==========================================

    async findAll(params: PaginationParams) {
        const where: Prisma.UserWhereInput = {};

        if (params.search) {
            where.OR = [
                { email: { contains: params.search, mode: 'insensitive' } },
                { name: { contains: params.search, mode: 'insensitive' } },
                { surname: { contains: params.search, mode: 'insensitive' } },
                { username: { contains: params.search, mode: 'insensitive' } },
                { phoneNumber: { contains: params.search, mode: 'insensitive' } },
            ];
        }

        return this.paginationService.paginate<any>(
            this.prisma.user,
            params,
            where,
        );
    }

    async findOneById(id: number) {
        // Try cache first
        const cacheKey = this.CACHE_KEYS.USER_PROFILE(id);
        const cached = await this.redis.get<any>(cacheKey);
        if (cached) {
            this.logger.debug(`Cache hit for user ${id}`);
            return cached;
        }

        const user = await this.prisma.user.findUnique({
            where: { id },
            select: {
                id: true,
                email: true,
                username: true,
                name: true,
                surname: true,
                role: true,
                status: true,
                phoneNumber: true,
                profilePhoto: true,
                profileDescription: true,
                gender: true,
                dateOfBirth: true,
                age: true,
                riskProfile: true,
                maxDailyLoss: true,
                maxPositionSize: true,
                maxLeverage: true,
                tradingEnabled: true,
                circuitBreakerUntil: true,
                emailVerified: true,
                lastLoginAt: true,
                createdAt: true,
                updatedAt: true,
            },
        });

        if (!user) {
            throw new NotFoundException('User not found');
        }

        // Cache for 5 minutes
        await this.redis.set(cacheKey, user, RedisService.TTL.MEDIUM);

        return user;
    }

    async findByUsername(username: string) {
        // Try cache first
        const cacheKey = this.CACHE_KEYS.USER_BY_USERNAME(username);
        const cached = await this.redis.get<any>(cacheKey);
        if (cached) {
            return cached;
        }

        const user = await this.prisma.user.findFirst({
            where: { username },
            select: {
                id: true,
                email: true,
                username: true,
                name: true,
                surname: true,
                role: true,
                status: true,
                profilePhoto: true,
                profileDescription: true,
                riskProfile: true,
                tradingEnabled: true,
                createdAt: true,
            },
        });

        if (!user) {
            throw new NotFoundException('User not found');
        }

        // Cache for 5 minutes
        await this.redis.set(cacheKey, user, RedisService.TTL.MEDIUM);

        return user;
    }

    // ==========================================
    // PROFILE UPDATE
    // ==========================================

    async updateProfile(userId: number, updateData: UpdateUserProfileDto) {
        const existingUser = await this.prisma.user.findUnique({
            where: { id: userId },
        });

        if (!existingUser) {
            throw new NotFoundException('User not found');
        }

        // Email uniqueness check
        if (updateData.email && updateData.email !== existingUser.email) {
            const emailExists = await this.prisma.user.findUnique({
                where: { email: updateData.email },
            });
            if (emailExists) {
                throw new ConflictException('Email already in use');
            }
        }

        // Username uniqueness check
        if (updateData.username && updateData.username !== existingUser.username) {
            const usernameExists = await this.prisma.user.findFirst({
                where: {
                    username: updateData.username,
                    NOT: { id: userId },
                },
            });
            if (usernameExists) {
                throw new ConflictException('Username already taken');
            }
        }

        // Phone uniqueness check
        if (updateData.phoneNumber && updateData.phoneNumber !== existingUser.phoneNumber) {
            const phoneExists = await this.prisma.user.findFirst({
                where: {
                    phoneNumber: updateData.phoneNumber,
                    NOT: { id: userId },
                },
            });
            if (phoneExists) {
                throw new ConflictException('Phone number already in use');
            }
        }

        const updatePayload: Prisma.UserUpdateInput = {};

        if (updateData.email !== undefined) updatePayload.email = updateData.email;
        if (updateData.username !== undefined) updatePayload.username = updateData.username;
        if (updateData.name !== undefined) updatePayload.name = updateData.name;
        if (updateData.surname !== undefined) updatePayload.surname = updateData.surname;
        if (updateData.phoneNumber !== undefined) updatePayload.phoneNumber = updateData.phoneNumber;
        if (updateData.gender !== undefined) updatePayload.gender = updateData.gender;
        if (updateData.profilePhoto !== undefined) updatePayload.profilePhoto = updateData.profilePhoto;
        if (updateData.profileDescription !== undefined) updatePayload.profileDescription = updateData.profileDescription;
        if (updateData.riskProfile !== undefined) updatePayload.riskProfile = updateData.riskProfile;
        if (updateData.dateOfBirth !== undefined) updatePayload.dateOfBirth = new Date(updateData.dateOfBirth);

        const updatedUser = await this.prisma.user.update({
            where: { id: userId },
            data: updatePayload,
            select: {
                id: true,
                email: true,
                username: true,
                name: true,
                surname: true,
                phoneNumber: true,
                profilePhoto: true,
                profileDescription: true,
                gender: true,
                riskProfile: true,
                tradingEnabled: true,
                emailVerified: true,
            },
        });

        // Invalidate cache
        await this.invalidateUserCache(userId, existingUser.username);

        this.logger.log(`Profile updated for user ${userId}`);

        return {
            message: 'Profile updated successfully',
            user: updatedUser,
        };
    }

    // ==========================================
    // USER TRADING STATS
    // ==========================================

    async getUserStats(userId: number): Promise<UserProfileStatsDto> {
        // Try cache first
        const cacheKey = this.CACHE_KEYS.USER_STATS(userId);
        const cached = await this.redis.get<UserProfileStatsDto>(cacheKey);
        if (cached) {
            this.logger.debug(`Cache hit for user stats ${userId}`);
            return cached;
        }

        // Get portfolio stats
        const portfolios = await this.prisma.portfolio.findMany({
            where: { userId },
            select: {
                totalValue: true,
                totalPnL: true,
            },
        });

        const totalPortfolioValue = portfolios.reduce(
            (sum, p) => sum + Number(p.totalValue),
            0,
        );
        const totalPnL = portfolios.reduce(
            (sum, p) => sum + Number(p.totalPnL),
            0,
        );

        // Get open positions count
        const openPositions = await this.prisma.position.count({
            where: {
                portfolio: { userId },
            },
        });

        // Get active alerts count
        const activeAlerts = await this.prisma.alert.count({
            where: { userId, isActive: true },
        });

        // Get AI signals count
        const totalAISignals = await this.prisma.aISignal.count({
            where: { userId },
        });

        // Get watchlist count
        const watchlistCount = await this.prisma.watchlist.count({
            where: { userId },
        });

        const stats: UserProfileStatsDto = {
            totalPortfolios: portfolios.length,
            totalPortfolioValue: totalPortfolioValue.toFixed(2),
            totalRealizedPnL: '0.00',
            totalUnrealizedPnL: totalPnL.toFixed(2),
            openPositions,
            activeAlerts,
            totalAISignals,
            watchlistCount,
        };

        // Cache for 1 minute (stats change frequently)
        await this.redis.set(cacheKey, stats, RedisService.TTL.SHORT);

        return stats;
    }

    // ==========================================
    // SEARCH USERS (Admin)
    // ==========================================

    async searchUsers(searchParams: SearchUserDto) {
        const {
            page = 1,
            limit = 10,
            searchTerm,
            dateFilter,
            sortBy = 'createdAt',
            sortDirection = 'desc',
            status,
            role,
            riskProfile,
            tradingEnabled,
            emailVerified,
            ...filters
        } = searchParams;

        const allowedSortFields = [
            'id', 'email', 'name', 'surname', 'username',
            'phoneNumber', 'createdAt', 'lastLoginAt',
        ];

        const sortField = allowedSortFields.includes(sortBy) ? sortBy : 'createdAt';
        const sortDir = ['asc', 'desc'].includes(sortDirection) ? sortDirection : 'desc';

        const andConditions: Prisma.UserWhereInput[] = [];

        // Basic filters
        if (filters.email) {
            andConditions.push({ email: { contains: filters.email, mode: 'insensitive' } });
        }
        if (filters.name) {
            andConditions.push({ name: { contains: filters.name, mode: 'insensitive' } });
        }
        if (filters.username) {
            andConditions.push({ username: { contains: filters.username, mode: 'insensitive' } });
        }

        // Enum filters
        if (status) andConditions.push({ status });
        if (role) andConditions.push({ role });
        if (riskProfile) andConditions.push({ riskProfile });
        if (tradingEnabled !== undefined) andConditions.push({ tradingEnabled });
        if (emailVerified !== undefined) andConditions.push({ emailVerified });

        // Search term filter
        if (searchTerm) {
            andConditions.push({
                OR: [
                    { email: { contains: searchTerm, mode: 'insensitive' } },
                    { name: { contains: searchTerm, mode: 'insensitive' } },
                    { surname: { contains: searchTerm, mode: 'insensitive' } },
                    { username: { contains: searchTerm, mode: 'insensitive' } },
                    { phoneNumber: { contains: searchTerm, mode: 'insensitive' } },
                ],
            });
        }

        // Date filter
        if (dateFilter) {
            const dateFilterQuery = this.createDateFilter(dateFilter);
            if (dateFilterQuery) andConditions.push(dateFilterQuery);
        }

        const where: Prisma.UserWhereInput = andConditions.length > 0
            ? { AND: andConditions }
            : {};

        const [users, total] = await Promise.all([
            this.prisma.user.findMany({
                skip: (page - 1) * limit,
                take: +limit,
                where,
                orderBy: { [sortField]: sortDir },
                select: {
                    id: true,
                    email: true,
                    username: true,
                    name: true,
                    surname: true,
                    role: true,
                    status: true,
                    profilePhoto: true,
                    riskProfile: true,
                    tradingEnabled: true,
                    emailVerified: true,
                    lastLoginAt: true,
                    createdAt: true,
                },
            }),
            this.prisma.user.count({ where }),
        ]);

        return {
            data: users,
            total,
            page: +page,
            limit: +limit,
            totalPages: Math.ceil(total / +limit),
        };
    }

    // ==========================================
    // ADMIN USER MANAGEMENT
    // ==========================================

    async adminUpdateUser(userId: number, updateData: AdminUpdateUserDto) {
        const user = await this.prisma.user.findUnique({
            where: { id: userId },
        });

        if (!user) {
            throw new NotFoundException('User not found');
        }

        const updatePayload: Prisma.UserUpdateInput = {};

        if (updateData.status !== undefined) updatePayload.status = updateData.status;
        if (updateData.role !== undefined) updatePayload.role = updateData.role;
        if (updateData.emailVerified !== undefined) updatePayload.emailVerified = updateData.emailVerified;
        if (updateData.tradingEnabled !== undefined) updatePayload.tradingEnabled = updateData.tradingEnabled;
        if (updateData.circuitBreakerUntil !== undefined) {
            updatePayload.circuitBreakerUntil = updateData.circuitBreakerUntil;
        }

        const updatedUser = await this.prisma.user.update({
            where: { id: userId },
            data: updatePayload,
        });

        // Invalidate cache
        await this.invalidateUserCache(userId, user.username);

        // Queue analytics job for admin actions
        await this.queue.addAnalyticsJob({
            type: 'risk',
            userId,
        });

        this.logger.log(`Admin updated user ${userId}: ${JSON.stringify(updateData)}`);

        return {
            message: 'User updated successfully',
            user: updatedUser,
        };
    }

    async toggleUserTrading(userId: number, enabled: boolean) {
        const user = await this.prisma.user.update({
            where: { id: userId },
            data: {
                tradingEnabled: enabled,
                circuitBreakerUntil: enabled ? null : undefined,
            },
        });

        // Invalidate cache
        await this.invalidateUserCache(userId, user.username);

        // Send notification
        await this.queue.addNotificationJob({
            userId,
            type: 'push',
            template: enabled ? 'trading_enabled' : 'trading_disabled',
            data: { tradingEnabled: enabled },
        });

        this.logger.log(`Trading ${enabled ? 'enabled' : 'disabled'} for user ${userId}`);

        return {
            message: `Trading ${enabled ? 'enabled' : 'disabled'} successfully`,
            tradingEnabled: user.tradingEnabled,
        };
    }

    // ==========================================
    // CACHE HELPERS
    // ==========================================

    private async invalidateUserCache(userId: number, username?: string | null) {
        await this.redis.delete(this.CACHE_KEYS.USER_PROFILE(userId));
        await this.redis.delete(this.CACHE_KEYS.USER_STATS(userId));
        if (username) {
            await this.redis.delete(this.CACHE_KEYS.USER_BY_USERNAME(username));
        }
        this.logger.debug(`Cache invalidated for user ${userId}`);
    }

    // ==========================================
    // HELPERS
    // ==========================================

    private createDateFilter(dateFilter: string): Prisma.UserWhereInput | undefined {
        const durations: Record<string, number> = {
            '24h': 1,
            '3d': 3,
            '7d': 7,
            '15d': 15,
            '30d': 30,
        };

        const days = durations[dateFilter];
        if (!days) return undefined;

        const startDate = new Date(Date.now() - days * 24 * 60 * 60 * 1000);

        return {
            createdAt: {
                gte: startDate,
                lte: new Date(),
            },
        };
    }
}