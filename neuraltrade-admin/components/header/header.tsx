'use client';

import { ActionIcon, Avatar, Flex, Group, Text, useMantineColorScheme, useMantineTheme } from "@mantine/core";
import { useSession } from "next-auth/react";
import { usePathname } from "next/navigation";
import classes from "./header.module.scss";
import { getPageTitle } from "@/utils/page-titles";
import { useState } from "react";


export function Header() {
    const pathname = usePathname();
    const { data: session } = useSession();
    const pageTitle = getPageTitle(pathname);
    const [error, setError] = useState(false);
    const { colorScheme, setColorScheme } = useMantineColorScheme();
    const theme = useMantineTheme();

    return (
        <Flex className={classes.header} pos="fixed" top={0} h={70}>
            <Text className={classes.pageTitle}>{pageTitle}</Text>
            <ActionIcon onClick={() => setColorScheme(colorScheme === 'dark' ? 'light' : 'dark')}>
                {colorScheme === 'dark' ? '☀️' : '🌙'}
            </ActionIcon>

            <Group>
                <Avatar
                    src={session?.user?.image}
                    w={45}
                    h={45}
                    radius="45px"
                    alt={session?.user?.username || 'User'}
                />
                <Flex direction="column" gap={3}>
                    <Text className={classes.username}>{session?.user?.username}</Text>
                    <Text className={classes.role}>
                    {session?.user?.role === 'SUPER_ADMIN' ? 'Süper Admin' : 'Admin'}
                    </Text>
                </Flex>
            </Group>
        </Flex>
    );
}