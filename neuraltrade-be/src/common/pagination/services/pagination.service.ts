import { Injectable } from '@nestjs/common';
// import { PrismaService } from '../../../core/prisma/prisma.service';
import { PaginationParams } from '../interfaces/pagination.interface';
// import { Prisma } from '@prisma/client';

@Injectable()
export class PaginationService {
  constructor() {}

  async paginate<T extends object>(
    model: any,
    params: PaginationParams,
    where: any = {},
    include: any = {},
  ) {
    const { page = 1, perPage = 10, sortBy, sortDirection = 'desc' } = params;

    const pageNumber = Number(page);
    const perPageNumber = Number(perPage);
    const skip = (pageNumber - 1) * perPageNumber;

    // orderBy option
    const orderBy: any = {};
    if (sortBy) {
      orderBy[sortBy] = sortDirection;
    } else {
      orderBy.createdAt = sortDirection;
    }

    // Total record count
    const total = await model.count({ where });

    // get datas
    const items: T[] = await model.findMany({
      skip,
      take: perPageNumber,
      where,
      orderBy,
      include,
    });

    // last page calculation
    const lastPage = Math.ceil(total / perPageNumber);

    return {
      items,
      meta: {
        total,
        page: pageNumber,
        lastPage,
        perPage: perPageNumber,
        prev: pageNumber > 1 ? pageNumber - 1 : null,
        next: pageNumber < lastPage ? pageNumber + 1 : null,
      },
    };
  }
}
