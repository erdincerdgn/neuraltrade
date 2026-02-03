import { IsString, IsNumber, IsOptional, IsEnum, Min } from 'class-validator';
import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';
import { OrderType, OrderSide } from '@prisma/client';

export class CreateOrderDto {
    @ApiProperty({ description: 'Trading symbol', example: 'AAPL' })
    @IsString()
    symbol: string;

    @ApiProperty({ enum: OrderSide, description: 'Order side' })
    @IsEnum(OrderSide)
    side: OrderSide;

    @ApiProperty({ enum: OrderType, description: 'Order type' })
    @IsEnum(OrderType)
    type: OrderType;

    @ApiProperty({ description: 'Quantity to trade', example: 100 })
    @IsNumber()
    @Min(0.00001)
    quantity: number;

    @ApiPropertyOptional({ description: 'Limit price (for LIMIT orders)' })
    @IsOptional()
    @IsNumber()
    price?: number;

    @ApiPropertyOptional({ description: 'Stop price (for STOP orders)' })
    @IsOptional()
    @IsNumber()
    stopPrice?: number;

    @ApiPropertyOptional({ description: 'Time in force', default: 'GTC' })
    @IsOptional()
    @IsString()
    timeInForce?: string;
}

export class ClosePositionDto {
    @ApiPropertyOptional({ enum: OrderType, description: 'Close order type', default: 'MARKET' })
    @IsOptional()
    @IsEnum(OrderType)
    type?: OrderType;

    @ApiPropertyOptional({ description: 'Quantity to close (partial close)' })
    @IsOptional()
    @IsNumber()
    quantity?: number;

    @ApiPropertyOptional({ description: 'Limit price (for LIMIT close)' })
    @IsOptional()
    @IsNumber()
    price?: number;
}
