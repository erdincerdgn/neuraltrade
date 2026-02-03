// import axios, { HttpStatusCode } from 'axios';
// import { getSession } from 'next-auth/react';
// import { notifications } from '@mantine/notifications';
// import { errorMessages } from '@/constants/errorMessages';

// const axiosWithToken = axios.create({
//   baseURL: process.env.NEXT_PUBLIC_API_URL,
//   headers: {
//     'Content-Type': 'application/json',
//   },
// });

// axiosWithToken.interceptors.request.use(async (config) => {
//   const session = await getSession();
//   if (session?.accessToken) {
//     config.headers.Authorization = `Bearer ${session.accessToken}`;
//   }
//   return config;
// });

// axiosWithToken.interceptors.response.use(
//   (response) => response,
//   (error) => {
//     const { response } = error;
//     if (response.status === HttpStatusCode.BadRequest && response.data?.message) {
//       const errorKey = response.data.message as keyof typeof errorMessages;
//       notifications.show({
//         title: 'Hata',
//         message: errorMessages[errorKey],
//         color: 'red',
//       });
//     }
//     return Promise.reject(error);
//   }
// );

// const axiosWithoutToken = axios.create({
//   baseURL: process.env.NEXT_PUBLIC_API_URL,
//   headers: {
//     'Content-Type': 'application/json',
//   },
// });

// const axiosWithTokenFileUpload = axios.create({
//   baseURL: process.env.NEXT_PUBLIC_API_URL,
//   headers: {
//     'Content-Type': 'multipart/form-data',
//   },
// });

// axiosWithTokenFileUpload.interceptors.request.use(async (config) => {
//   const session = await getSession();
//   if (session?.user.accessToken) {
//     // eslint-disable-next-line no-param-reassign
//     config.headers.Authorization = `Bearer ${session.accessToken}`;
//   }
//   return config;
// });

// export { axiosWithToken, axiosWithoutToken, axiosWithTokenFileUpload };
