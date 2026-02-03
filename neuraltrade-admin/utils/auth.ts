import { AuthOptions } from 'next-auth';
import CredentialsProvider from 'next-auth/providers/credentials';
import { login } from './api/user';

export const authOptions: AuthOptions = {
  session: {
    strategy: 'jwt',
  },
  secret: process.env.NEXTAUTH_SECRET,
  pages: {
    signIn: '/login', // updated to match new login route
  },
  providers: [
    CredentialsProvider({
      name: 'Credentials',
      credentials: {
        email: { label: 'Email', type: 'email', placeholder: 'test@mail.com' },
        password: { label: 'Password', type: 'password' },
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) {
          return null;
        }
        try {
          const data = await login(credentials.email, credentials.password);
          return {
            id: data.user.id,
            email: data.user.email,
            username: data.user.username,
            name: data.user.name,
            role: data.user.role,
            image: data.user.profilePhoto ?? null,
            accessToken: data.accessToken,
          };
        } catch (error: any) {
          if (error?.response?.status === 401) {
            throw new Error('INVALID_CREDENTIALS');
          }
          throw new Error('SERVER_ERROR');
        }
      },
    }),
  ],
  callbacks: {
    async signIn({ user }) {
      const u = user as any;
      const role = u.role;
      // const company = u.company ?? u.companyName ?? null;
      
      // Only allow SUPER_ADMIN or NeuralTrade company users
      if (role === 'SUPER_ADMIN') {
        return true;
      }
      // if (company === 'NeuralTrade') {
      //   return true;
      // }
      
      // Block everyone else
      return false;
    },
    async jwt({ token, user }) {
      if (user) {
        return {
          ...token,
          username: user.username,
          role: user.role,
          image: user.image,
          accessToken: user.accessToken,
          // company: (user as any).company ?? (user as any).companyName ?? null,
        };
      }
      return token;
    },
    async session({ session, token }) {
      return {
        ...session,
        user: {
          ...session.user,
          id: token.sub,
          username: token.username,
          role: token.role,
          image: token.image as string | null,
          // company: (token as any).company ?? null,
        },
        accessToken: token.accessToken,
      };
    },
  },
};
