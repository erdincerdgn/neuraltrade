import { useState, useEffect } from 'react';
import axios, { Method } from 'axios';

interface FetchOptions {
  method?: Method;
  body?: any;
  headers?: Record<string, string>;
}

interface FetchResult<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

export function useFetch<T = any>(url: string, options: FetchOptions = {}): FetchResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const { method = 'GET', body = null, headers = {} } = options;

  useEffect(() => {
    const fetchData = async () => {
      try {
        const result = await axios({
          url,
          method,
          data: body,
          headers,
        });
        setData(result.data);
      } catch (err) {
        if (axios.isAxiosError(err) && err.message) {
          setError(err.message);
        } else {
          setError('Error');
        }
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [url, method]);

  return { data, loading, error };
}
