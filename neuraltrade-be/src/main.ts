import { NestFactory } from '@nestjs/core';
import { NestFastifyApplication, FastifyAdapter } from '@nestjs/platform-fastify';
import { AppModule } from './app.module';
import fastifyCors from '@fastify/cors';
import fastifyHelmet from '@fastify/helmet';
import fastifyMultipart from '@fastify/multipart';
import fastifyCompress from '@fastify/compress';
import fastifyRateLimit from '@fastify/rate-limit';
import { DocumentBuilder, SwaggerModule } from '@nestjs/swagger';
import { ValidationPipe, Logger, VersioningType } from '@nestjs/common';
import { RedisIoAdapter } from './adapters/redis-io.adapter';

async function bootstrap() {
    const logger = new Logger('Bootstrap');
    const isProduction = process.env.NODE_ENV === 'production';

    // Create Fastify application
    const app = await NestFactory.create<NestFastifyApplication>(
        AppModule,
        new FastifyAdapter({
            logger: !isProduction, // Disable Fastify logger in production (use NestJS logger)
            trustProxy: true,
            bodyLimit: 50 * 1024 * 1024, // 50 MB
        }),
    );

    // ==========================================
    // API Versioning
    // ==========================================
    app.enableVersioning({
        type: VersioningType.URI,
        defaultVersion: '1',
        prefix: 'api/v',
    });

    // Global prefix for all routes (except health checks)
    app.setGlobalPrefix('api', {
        exclude: ['/', 'health', 'health/ready', 'health/live', 'docs'],
    });

    // ==========================================
    // Fastify Plugins
    // ==========================================

    // File uploads
    await app.register(fastifyMultipart, {
        limits: {
            fileSize: 10 * 1024 * 1024, // 10 MB for uploads
            files: 5,
            fieldSize: 1 * 1024 * 1024,
            fieldNameSize: 255,
            headerPairs: 2000,
        },
    });

    // CORS configuration
    const allowedOrigins = process.env.CORS_ORIGINS?.split(',') || ['*'];
    await app.register(fastifyCors, {
        origin: isProduction ? allowedOrigins : '*',
        methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'],
        allowedHeaders: ['Content-Type', 'Authorization', 'X-Request-ID'],
        credentials: true,
        maxAge: 86400, // 24 hours
    });

    // Security headers
    await app.register(fastifyHelmet, {
        contentSecurityPolicy: isProduction ? undefined : false, // Disable CSP for Swagger in dev
        crossOriginEmbedderPolicy: false,
    });

    // Compression
    await app.register(fastifyCompress, {
        encodings: ['gzip', 'deflate'],
    });

    // Rate limiting
    await app.register(fastifyRateLimit, {
        max: isProduction ? 100 : 1000, // requests per window
        timeWindow: '1 minute',
        errorResponseBuilder: () => ({
            statusCode: 429,
            error: 'Too Many Requests',
            message: 'Rate limit exceeded. Please try again later.',
        }),
    });

    // <--- YENİ EKLENDİ: Burası WebSocket hatasını çözen kritik kısımdır
    try {
        const redisIoAdapter = new RedisIoAdapter(app);
        await redisIoAdapter.connectToRedis(); // Redis bağlantısını bekle
        app.useWebSocketAdapter(redisIoAdapter); // Uygulamaya adapter'ı tanıt
        logger.log('🔌 Redis WebSocket Adapter initialized');
    } catch (e) {
        logger.warn('⚠️ Redis adapter failed to connect, falling back to default (memory) adapter:', e);
    }

    // ==========================================
    // Swagger Documentation (disable in production if needed)
    // ==========================================
    if (!isProduction || process.env.ENABLE_SWAGGER === 'true') {
        const config = new DocumentBuilder()
            .setTitle('NeuralTrade API')
            .setDescription(`
        **NeuralTrade Trading Platform API**
        
        AI-powered trading platform with:
        - Portfolio Management
        - AI Trading Signals
        - Risk Management
        - Real-time Market Data
        
        **Authentication:** Use JWT Bearer token
      `)
            .setVersion('1.0.0')
            .setContact('NeuralTrade Team', 'https://neuraltrade.io', 'support@neuraltrade.io')
            .addServer(process.env.API_URL || 'http://localhost:3000', 'API Server')
            .addBearerAuth(
                {
                    type: 'http',
                    scheme: 'bearer',
                    bearerFormat: 'JWT',
                    description: 'Enter your JWT token',
                },
                'JWT',
            )
            .addTag('System', 'Health checks and system info')
            .addTag('Auth', 'Authentication and authorization')
            .addTag('Users', 'User management')
            .build();

        const document = SwaggerModule.createDocument(app, config, {
            deepScanRoutes: true,
            operationIdFactory: (_controllerKey: string, methodKey: string) => methodKey,
        });

        SwaggerModule.setup('docs', app, document, {
            useGlobalPrefix: false, // yeni eklendi
            swaggerOptions: {
                persistAuthorization: true,
                defaultModelsExpandDepth: 1,
                defaultModelExpandDepth: 1,
                docExpansion: 'list',
                filter: true,
                showRequestDuration: true,
            },
            customSiteTitle: 'NeuralTrade API Docs',
        });

        logger.log('📚 Swagger documentation enabled at /docs');
    }

    // ==========================================
    // Global Pipes, Filters, Interceptors
    // ==========================================

    // Validation pipe
    app.useGlobalPipes(
        new ValidationPipe({
            transform: true,
            whitelist: true,
            forbidNonWhitelisted: true,
            transformOptions: {
                enableImplicitConversion: true,
            },
            enableDebugMessages: !isProduction,
            disableErrorMessages: isProduction,
        }),
    );

    // Note: Global filters and interceptors are registered in CommonModule via APP_FILTER/APP_INTERCEPTOR

    // ==========================================
    // Graceful Shutdown
    // ==========================================
    app.enableShutdownHooks();

    // Handle uncaught exceptions
    process.on('uncaughtException', (error) => {
        logger.error('Uncaught Exception:', error);
    });

    process.on('unhandledRejection', (reason) => {
        logger.error('Unhandled Rejection:', reason);
    });

    // ==========================================
    // Start Server
    // ==========================================
    const port = parseInt(process.env.PORT || '3000', 10);
    const host = process.env.HOST || '0.0.0.0';

    await app.listen(port, host);

    logger.log(`🚀 NeuralTrade API running on: ${await app.getUrl()}`);
    logger.log(`📊 Environment: ${process.env.NODE_ENV || 'development'}`);

    if (!isProduction) {
        logger.log(`📚 Swagger docs: ${await app.getUrl()}/docs`);
        logger.log(`❤️ Health check: ${await app.getUrl()}/health`);
    }
}

bootstrap().catch((error) => {
    console.error('❌ Failed to start application:', error);
    process.exit(1);
});