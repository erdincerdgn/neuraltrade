import dayjs from 'dayjs';

export function addSevenDaysToNow() {
  const now = dayjs();
  const newDate = now.add(7, 'day');
  return newDate.toISOString();
}

export const formatDate = (date: string | Date): string =>
  new Date(date).toLocaleDateString('tr-TR', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
  });

  export const formatDateTime = (date: string | Date): string =>
  new Date(date).toLocaleDateString('tr-TR', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
