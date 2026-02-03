// 'use client';

// import { ReactNode, useEffect } from 'react';
// import { useRouter } from 'next/navigation';
// import { useSession } from 'next-auth/react';

// const RequireAuth: React.FC<{ children: ReactNode }> = ({ children }) => {
//   const { data: session, status } = useSession();
//   const router = useRouter();

//   useEffect(() => {
//     if (status === 'unauthenticated') {
//       router.push('/auth/signin');
//     }
//   }, [status, router]);

//   if (status === 'loading') return <p>Loading...</p>;

//   return <>{session ? children : null}</>;
// };

// export default RequireAuth;
