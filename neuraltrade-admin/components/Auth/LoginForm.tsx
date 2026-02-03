'use client';

import { useEffect, useState } from 'react';
import { TextInput, PasswordInput, Checkbox, Button, Card, Stack, Image, Center, Paper, Title, Group,Text, useMantineColorScheme, Anchor, useMantineTheme, ActionIcon } from '@mantine/core';
import { useRouter } from 'next/navigation';
import { useForm } from '@mantine/form';
import { signIn, signOut } from 'next-auth/react';
import { notificationError } from '@/utils/notification-manager';
import classes from './login-form.module.scss'
import { IconBuilding } from '@tabler/icons-react';

interface ILoginForm {
  email: string;
  password: string;
  rememberMe: boolean;
}

export function LoginForm() {
  const router = useRouter();
  const [error, setError] = useState(false);
  const [mounted, setMounted] = useState(false);
  const { colorScheme, setColorScheme } = useMantineColorScheme();
  // const dark = colorScheme === 'dark';
  const theme = useMantineTheme();

  const form = useForm({
    initialValues: {
      email: '',
      password: '',
      rememberMe: false,
    },

    validate: {
      email: (value) => (/^\S+@\S+$/.test(value) ? null : 'Invalid email'),
      password: (value) => (value.length < 1 ? 'Password is required' : null),
    },
  });

  const onSubmit = (val: ILoginForm) => {
    signIn('credentials', {
      email: val.email,
      password: val.password,
      redirect: false,
    }).then((signInData) => {
      if (signInData?.error) {
        switch (signInData.error) {
          case 'INVALID_CREDENTIALS':
            notificationError({ message: 'Username or password is incorrect' });
            break;
          case 'SERVER_ERROR':
            notificationError({ message: 'Server error, please try again later' });
            break;
          case 'AccessDenied':
            notificationError({ message: 'Only NeuralTrade or SUPER_ADMIN users can access the admin panel.' });
            break;
          default:
            notificationError({
              message: 'Login failed',
            });
            break;
        }
      } else {
        if (val.rememberMe) {
          localStorage.setItem('rememberedEmail', val.email);
          localStorage.setItem('rememberedPassword', val.password);
        } else {
          localStorage.removeItem('rememberedEmail');
          localStorage.removeItem('rememberedPassword');
        }

        router.refresh();
        router.push('/dashboard');
      }
    });
  };

  useEffect(() => {
    setMounted(true);
    
    const rememberedEmail = localStorage.getItem('rememberedEmail');
    const rememberedPassword = localStorage.getItem('rememberedPassword');

    if (rememberedEmail && rememberedPassword) {
      form.setValues({
        email: rememberedEmail,
        password: rememberedPassword,
        rememberMe: true,
      });
    }
  }, []);

  return (

    <Center mih="100vh" w="100vw">
      {mounted && (
        <ActionIcon onClick={() => setColorScheme(colorScheme === 'dark' ? 'light' : 'dark')}>
          {colorScheme === 'dark' ? '☀️' : '🌙'}
        </ActionIcon>
      )}
      <Paper
        p="xl"
        shadow="md"
        w="100%"
        maw={670}
        miw={360}
        component="form"
        onSubmit={form.onSubmit(onSubmit)}
        
        
      >
        <Stack align='center' mb="lg" gap={4}>
          <Group gap="xs">
            <IconBuilding size={32} />
            <Title order={2}>
              NeuralTrade Admin
            </Title>
          </Group>
          <Text size="sm">
            Secure login for the NeuralTrade team.
          </Text>
        </Stack>

        <Stack gap="md">
          <div>
            <Text size="sm" fw={500} mb={4}>
              Username/Email
            </Text>
            <TextInput
              placeholder="you@example.com"
              radius="lg"
              required
              type="email"
              {...form.getInputProps('email')}
              
            />
          </div>

          <div>
            <Text size="sm" fw={500} mb={4}>
              Password
            </Text>
            <PasswordInput
              placeholder="Enter your password"
              radius="lg"
              required
              {...form.getInputProps('password')}
              
            />
          </div>

          <Group justify="space-between" mt="xs">
            <Checkbox
              label="Remember me"
              size="sm"
              {...form.getInputProps('rememberMe', { type: 'checkbox' })}
            />
            <Anchor size="sm" fw={500}>
              Forgot Password?
            </Anchor>
          </Group>

          <Button
            type="submit"
            fullWidth
            radius="lg"
            
            size="sm"
            mt="xs"
          
            onMouseDown={(e) => e.preventDefault()}
          >
            Login
          </Button>

          {error && (
            <Text
              size="sm"
              ta="center"
              
              mt="xs"
            >
              Incorrect username or password.
            </Text>
          )}
        </Stack>

      </Paper>
    </Center>





    // <Card withBorder shadow="md" p={30} mt={250} radius="md" w={500}>
    //   <Stack align='center' mb={20}>
    //     <Image
    //       src="/icons/logo.svg"
    //       alt="App Logo"
    //       width={120}
    //     />
    //   </Stack>
    //   <form onSubmit={form.onSubmit(onSubmit)}>
    //     <TextInput
    //       label="Email"
    //       placeholder="test@example.com"
    //       required
    //       {...form.getInputProps('email')}
    //     />
    //     <PasswordInput
    //       label="Password"
    //       placeholder="Your password"
    //       required
    //       mt="md"
    //       {...form.getInputProps('password')}
    //     />
    //     <Checkbox
    //       mt="lg"
    //       label="Remember me"
    //       checked={form.values.rememberMe}
    //       onChange={(event) => form.setFieldValue('rememberMe', event.currentTarget.checked)}
    //     />
    //     <Button type="submit" fullWidth mt="xl">
    //       Sign In
    //     </Button>
    //   </form>
    // </Card>
  );
}
