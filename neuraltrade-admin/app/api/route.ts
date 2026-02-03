import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth/next';
import { authOptions } from '@/utils/auth';

export const GET = async () => {
  const session = await getServerSession(authOptions);
  return NextResponse.json({ authenticated: !!session });
};
