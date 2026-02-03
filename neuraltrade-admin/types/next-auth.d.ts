// eslint-disable-next-line @typescript-eslint/no-unused-vars
import 'next-auth';

declare module 'next-auth' {
  interface User {
    username: string;
    role: string;
    accessToken: string;
    // company?: string | null;
    // companyName?: string | null;
  }

  interface Session {
    user: User & {
      id: string;
      access: string;
      // company?: string | null;
      // companyName?: string | null;
    };
    accessToken: string;
  }
}

declare module 'next-auth/jwt' {
  interface JWT {
    username: string;
    role: string;
    accessToken: string;
    // company?: string | null;
    // companyName?: string | null;
  }
}
