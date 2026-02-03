import { IsString, IsNumber, IsOptional, IsBoolean, Min } from 'class-validator';
import { ApiProperty, ApiPropertyOptional, PartialType } from '@nestjs/swagger';

export class CreatePortfolioDto {
    @ApiProperty({ description: 'Portfolio name', example: 'My Trading Portfolio' })
    @IsString()
    name: string;

    @ApiPropertyOptional({ description: 'Portfolio description' })
    @IsOptional()
    @IsString()
    description?: string;

    @ApiPropertyOptional({ description: 'Base currency', default: 'USD' })
    @IsOptional()
    @IsString()
    currency?: string;

    @ApiPropertyOptional({ description: 'Initial capital' })
    @IsOptional()
    @IsNumber()
    @Min(0)
    initialCapital?: number;

    @ApiPropertyOptional({ description: 'Set as default portfolio' })
    @IsOptional()
    @IsBoolean()
    isDefault?: boolean;
}

export class UpdatePortfolioDto extends PartialType(CreatePortfolioDto) { }
