import { ComboboxItem } from '@mantine/core';

/**
 * Bir tip tanımından Mantine Select için options dizisi oluşturur
 * 
 * @param values - Tip değerlerinin dizisi
 * @returns Select bileşeninin data prop'u için hazır dizi
 */
export function createSelectOptionsFromType<T extends string>(values: T[]): ComboboxItem[] {
  return values.map(value => ({
    value,
    label: value
  }));
}