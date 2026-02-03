'use client';

import { useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import { Collapse, Flex, NavLink, Text } from "@mantine/core";
import classes from "./styles.module.scss";
import { signOut, useSession } from "next-auth/react";
import Image from "next/image";
import { getNavbarList } from "@/constants/navbar-list";

export function Navbar() {
    const router = useRouter();
    const pathname = usePathname();
    const { data: session } = useSession();

    const [opened, setOpened] = useState<String[]>([]);
    const navbarList = getNavbarList();

    const toogleNested = (path: string) => {
        setOpened((current) => 
            current.includes(path) ? current.filter((item) => item !== path) : [...current, path]
        );
    };

    const handleItemClick = (item: any) => {
        if (item.key && item.key === 'logout') {
            signOut();
            router.replace('/login');
            return;
        }
        if (item.isNested) {
            toogleNested(item.path);
        } else {
            router.push(item.path);
        }
    };

    const handleSubItemClick = (subItem: any, parentItem: any) => {
        router.push(subItem.path);
        if (!opened.includes(parentItem.path)) {
            toogleNested(parentItem.path);
        }
    };

    const isItemActive = (item: any) => {
        if (item.isNested) {
            return (
                item.subItems.some((subItem: any) => pathname.startsWith(subItem.path)) || pathname === item.path
            );
        }
        return pathname === item.path;
    };

    const renderNavItems = navbarList.map((item) => (
        <div className="classes.navItem" key={item.label}>
            <NavLink
                label={item.label}
                // leftSection={isItemActive(item) ? item.activeIcon : item.inactiveIcon}
                onClick={() => handleItemClick(item)}
                active={isItemActive(item)}
                className={classes.navLink}
                style={{
                    padding: '0 0 0 18px',
                }}
            />
            {item.isNested && (
                <Collapse in={opened.includes(item.path)}>
                    {item.subItems.map((subItem) => (
                        <NavLink
                            key={subItem.label}
                            label={subItem.label}
                            // leftSection={
                            //     <div className={pathname === subItem.path ? classes.dotActive : classes.dot}/>
                            // }
                            pl={40}
                            onClick={() => handleSubItemClick(subItem, item)}
                            active={pathname === subItem.path}
                            // className={classes.subNavLink}
                        />
                    ))}
                </Collapse>
            )}
        </div>
    ));

    return (
        <nav className={classes.navbar}>
            <div className={classes.headerContainer}>
                <div className={classes.logoContainer}>
                    <Image src="/icons/logo-navbar.svg" alt="Logo" width={200} height={40}/>
                </div>
            </div>
            <Flex className={classes.dashboardTitleContainer}>
                <Text className={classes.dashboardTitle}>Welcome {session?.user?.name}, you're in control! 🔑</Text>
            </Flex>
            <div className={classes.navbarMain}>{renderNavItems}</div>
        </nav>
    )
}