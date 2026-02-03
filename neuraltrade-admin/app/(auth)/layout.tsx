'use client';

import { Center, Container } from '@mantine/core';

interface Props {
  children: React.ReactNode;
}

export default function AuthLayout({ children }: Props) {
  return (
    <Center>
      <Container size="xl">{children}</Container>
    </Center>
  );
}
