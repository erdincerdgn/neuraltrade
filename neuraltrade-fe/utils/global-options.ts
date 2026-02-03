export const getLabelByValue = (value: any, options: { value: any; label: string }[]) =>
    options.find((option) => option.value === value)?.label || '';