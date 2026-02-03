import { NextRequest, NextResponse } from 'next/server';
import { getToken } from 'next-auth/jwt';

export async function middleware(request: NextRequest) {
  const pathname = request.nextUrl.pathname;
  
  // Public routes that don't require authentication
  const publicRoutes = ['/login'];
  const isPublicRoute = publicRoutes.some(route => pathname.startsWith(route));
  
  // Skip auth check for public routes
  if (isPublicRoute) {
    return NextResponse.next();
  }
  
  // Protected admin routes
  const adminRoutes = ['/dashboard', '/listing', '/user', '/settings'];
  const isAdminRoute = adminRoutes.some(route => pathname.startsWith(route));
  
  if (!isAdminRoute) {
    return NextResponse.next();
  }

  const token = await getToken({ req: request, secret: process.env.NEXTAUTH_SECRET });
  
  // No token = not authenticated, just redirect (don't clear cookies)
  if (!token) {
    return NextResponse.redirect(new URL('/login', request.url));
  }
  
  const role = (token as any).role;
  // const company = (token as any).company ?? (token as any).companyName ?? null;
  
  // Strict check: only SUPER_ADMIN or NeuralTrade
  const isSuperAdmin = role === 'SUPER_ADMIN';
  // const isNeuralTrade = company === 'NeuralTrade';
  
  if (!isSuperAdmin) {
    // User is authenticated but NOT authorized - clear session
    const response = NextResponse.redirect(new URL('/login', request.url));
    
    // Clear NextAuth session cookies only for unauthorized users
    response.cookies.set('next-auth.session-token', '', { maxAge: 0 });
    response.cookies.set('__Secure-next-auth.session-token', '', { maxAge: 0 });
    
    return response;
  }
  
  // User is authorized, allow access
  return NextResponse.next();
}

export const config = {
  matcher: ['/dashboard/:path*', '/listing/:path*', '/user/:path*', '/settings/:path*'],
};
