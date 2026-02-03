import axios from 'axios';
import { getSession } from 'next-auth/react';

const axiosWithToken = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

axiosWithToken.interceptors.request.use(async (config) => {
  const session = await getSession();
  if (session?.accessToken) {
    config.headers.Authorization = `Bearer ${session.accessToken}`;
  }
  return config;
});

const axiosWithoutToken = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});
axiosWithoutToken.interceptors.request.use(async (config) => config);

export { axiosWithToken, axiosWithoutToken };
