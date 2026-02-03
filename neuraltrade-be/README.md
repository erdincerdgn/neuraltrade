# NeuralTrade Backend API

NestJS backend for NeuralTrade trading platform with AI integration.

## 🚀 Quick Start

```bash
npm install
npx prisma generate
npx prisma migrate dev
npm run start:dev
```

API will be available at: http://localhost:4000

## 📋 Features

- RESTful API + WebSocket
- Trading bot integration (CCXT)
- Job queue (BullMQ)
- Real-time market data
- AI service integration
- Authentication & Authorization

## 🔧 Tech Stack

- NestJS 11
- Prisma ORM
- PostgreSQL
- Redis (cache & pub/sub)
- Socket.IO
- CCXT (crypto exchanges)
- BullMQ (job queue)

## 🌐 Integrations

- Python AI Service: `http://localhost:8000`
- Qdrant Vector DB: `http://localhost:6333`
- Redis: `localhost:6379`
- Prometheus: `http://localhost:9091`

## 📝 Scripts

- `npm run start:dev` - Development server
- `npm run build` - Production build
- `npm run start:prod` - Start production
- `npm run prisma:migrate` - Run migrations
- `npm run prisma:studio` - Prisma Studio GUI
- `npm run test` - Run tests

## 📊 API Documentation

Swagger UI: http://localhost:4000/api
