// import { login } from "@/services/api/user";
// import { AuthOptions } from "next-auth";
// import CredentialsProvider from "next-auth/providers/credentials";

// export const authOptions: AuthOptions = {
//     session: {
//         strategy: "jwt",
//     },
//     secret: process.env.NEXTAUTH_SECRET,
//     pages: {
//         signIn: '/auth/signin',
//     },
//     providers: [
//         CredentialsProvider({
//             name: 'Credentials',
//             credentials: {
//                 email: { label: 'Email', type: 'email', placeholder: 'test@mail.com' },
//                 password: { label: 'Password', type: 'password' },
//             },
//             async authorize(credentials) {
//                 if (!credentials?.email || !credentials?.password) {
//                     return null;
//                 }
//                 try {
//                     const data = await login(credentials.email, credentials.password);

//                     return {
//                         id: data.user.id,
//                         email: data.user.email,
//                         username: data.user.username,
//                         name: data.user.name,
//                         surname: data.user.surname,
//                         role: data.user.role,
//                         profilePhoto: data.user.profilePhoto,
//                         officeId: data.user.officeId,
//                         companyName: data.user.companyName,
//                         phoneNumber: data.user.phoneNumber,
//                         officeLogo: data.user.officeLogo,
//                         accessToken: data.accessToken,
//                     };
//                 } catch (error) {
//                     return null;
//                 }
//             },
//         }),
//     ],
//     callbacks: {
//         async jwt({ token, user, session, trigger }) {
//             if (trigger === 'update' && session) {
//                 return {
//                     ...token,
//                     companyName: session.user.companyName,
//                     officeLogo: session.user.officeLogo,
//                 };
//             }

//             if (user) {
//                 return {
//                     ...token,
//                     id: user.id,
//                     email: user.email,
//                     username: user.username,
//                     name: user.name,
//                     surname: user.surname,
//                     role: user.role,
//                     profilePhoto: user.profilePhoto,
//                     officeId: user.officeId,
//                     companyName: user.companyName,
//                     companyLogo: user.officeLogo,
//                     phoneNumber: user.phoneNumber,
//                     accessToken: user.accessToken,
//                 };
//             }
//             return token;
//         },
//         async session({ session, token }) {
//             return {
//                 ...session,
//                 user: {
//                     id: token.id,
//                     email: token.email,
//                     username: token.username,
//                     name: token.name,
//                     surname: token.surname,
//                     role: token.role,
//                     profilePhoto: token.profilePhoto,
//                     officeId: token.officeId,
//                     companyName: token.companyName,
//                     officeLogo: token.officeLogo,
//                     phoneNumber: token.phoneNumber,
//                 },
//                 accessToken: token.accessToken,
//             };
//         },
//     },
// };