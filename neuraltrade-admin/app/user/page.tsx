'use client';

import { notificationError } from "@/utils/notification-manager";
import { Flex, Paper, TextInput, Text, Group, ActionIcon, Box } from "@mantine/core";
import { use, useEffect, useState } from "react";
import styles from "./user.module.scss";
import { IUser } from "@/types/user";
import { fetchUserApi, getUserDetails } from "@/utils/api/user";
import { DataTable, DataTableColumn } from "mantine-datatable";
import { IconEdit, IconEye, IconTrash } from "@tabler/icons-react";
import { useDisclosure } from "@mantine/hooks";
import DeleteUserModal from "@/components/user-status/delete-user.modal";
import EditUserModal from "@/components/user-status/edit-modal/edit-user-modal";

export default function User() {
    const [page, setPage] = useState(1);
    const [perPage, setPerPage] = useState(10);
    const [data, setData] = useState<{ items: IUser[]; meta: { total: number }}>({
        items: [],
        meta: { total: 0 },
    });
    const [isLoading, setIsLoading] = useState(false);
    const [currentUser, setCurrentUser] = useState<Partial<IUser | null>>(null);
    const [isEditing, setIsEditing] = useState(false);
    const [opened, {open, close}] = useDisclosure(false);
    const [openedApproveModal, {open: openApproveModal, close: closeApproveModal}] =  useDisclosure(false);
    const [isApproved, setIsApproved] = useState(false);

    const [currentUserId, setCurrentUserId] = useState<string>('');
    const [deleteModalOpened, setDeleteModalOpened] = useState(false);
    const [detailEditModalOpened, setDetailEditModalOpened] = useState(false);
    const [expandedRecordIds, setExpandedRecordIds] = useState<string[]>([]);

    const [searchTerm, setSearchTerm] = useState('');
    const [sortBy, setSortBy] = useState<string | undefined>(undefined);
    const [sortDirection, setSortDirection] = useState<'asc' | 'desc' | undefined>(undefined);

    const fetchData = async () => {
        try {
            setIsLoading(true);

            const allowedSortFields = ['id', 'email', 'name', 'surname', 'username', 'phoneNumber', 'createdAt'] as const;
            type SortField = typeof allowedSortFields[number];

            const validatedSortBy = allowedSortFields.includes(sortBy as SortField)
             ? (sortBy as SortField)
             : undefined;

            const response = await fetchUserApi({
                page,
                limit: perPage,
                searchTerm,
                sortBy: validatedSortBy,
                sortDirection,
            });
            setData(response);
        } catch (error) {
            console.log(error);
            notificationError({message: 'Failed to fetch users' });
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        fetchData();
    }, [page, perPage, searchTerm, sortBy, sortDirection]);

    useEffect(() => {
        setPage(1);
    }, [searchTerm, sortBy, sortDirection]);

    const handleRowExpansion = (id: string) => {
        setExpandedRecordIds((prevIds) => {
            if (prevIds.includes(id)) {
                return prevIds.filter(recordId => recordId !== id);
            } else {
                return [...prevIds, id];
            }
        });
    };

    const ExpandedRow = ({ record }: { record: IUser }) => (
        <Box p="md" bg="gray.0">
              <Flex direction={{ base: 'column', md: 'row' }} gap="md">
                <Box style={{ flex: 1, minWidth: '300px' }}>
                  
                </Box>
              </Flex>
        </Box>
    );

    const handleUserAction = async (id: string, isEdit: boolean) => {
        try {
            const response = await getUserDetails(id);
            if (response) {
              setCurrentUser({
                id: response.id,
                email: response.email,
                name: response.name,
                surname: response.surname,
                username: response.username,
                phoneNumber: response.phoneNumber,
              });
            }
        } catch (error) {
            notificationError({ message: 'Kullanıcı detayları yüklenirken bir hata oluştu' });
        }
        setIsEditing(isEdit);
        open();
    }

    const handleAllDetailEdit =  (id: string) => {
        setCurrentUserId(id);
        setDetailEditModalOpened(true);
    }

    const handleDeleteUser = (id: string) => {
        setCurrentUserId(id);
        setDeleteModalOpened(true);
    }

    const handleApproveUser = (id: string, approve: boolean) => {
        setCurrentUserId(id);
        setIsApproved(approve);
        openApproveModal();
    }

    const expansionColumn: DataTableColumn<IUser> = {
        accessor: 'expansion',
        title: '',
        width: 10,
    };

    const baseColumns: DataTableColumn<IUser>[] = [
        expansionColumn,
        {
            accessor: 'id',
            title: 'ID',
            sortable: true,
            width: 100,
            render: (record) => (
                <Text truncate className={styles.idText}>
                    {record.id}
                </Text>
            ),
        },
        {
            accessor:'email',
            title: 'Email',
            sortable: true,
            width: 100,
            render: (record) => (
                <Text truncate className={styles.idText}>
                {record.email}
                </Text>
            ),
        },
        {
            accessor: 'name',
            title: 'Name',
            sortable: true,
            width: 100,
            render: (record) => (
                <Text truncate className={styles.idText}>
                {record.name}
                </Text>
            ),
        },
        {
            accessor: 'surname',
            title: 'Surname',
            sortable: true,
            width: 100,
            render: (record) => (
                <Text truncate className={styles.idText}>
                {record.surname}
                </Text>
            ),
        },
        {
            accessor: 'username',
            title: 'Username',
            sortable: true,
            width: 105,
            render: (record) => (
                <Text truncate className={styles.idText}>
                    {record.username}
                </Text>
            ),
        },
        {
            accessor: 'phoneNumber',
            title: 'Phone Number',
            sortable: true,
            width: 110,
            render: (record) => (
                <Text truncate className={styles.idText}>
                {record.phoneNumber}
                </Text>
            ),
        },
        {
            accessor: 'actions',
            title: 'Eylemler',
            width: 110,
            render: (val: IUser) => (
                <Group className={styles.actionGroup}>
                <ActionIcon variant="default">
                    <IconEdit
                    size={20}
                    className={`${styles.actionButton} ${styles.edit}`}
                    onClick={() => status === 'PendingApproval' ?  handleAllDetailEdit(val.id) : handleUserAction(val.id, true)}
                    />
                </ActionIcon>
                <ActionIcon variant="default">
                    <IconTrash
                    size={20}
                    className={`${styles.actionButton} ${styles.delete}`}
                    onClick={() => handleDeleteUser(val.id)}
                    />
                </ActionIcon>
                <ActionIcon variant="default">
                    <IconEye
                    size={20}
                    className={`${styles.actionButton} ${styles.view}`}
                    onClick={() => handleUserAction(val.id, false)}
                    />
                </ActionIcon>
                </Group>
            ),
        },
    ];


    const columns = [...baseColumns];

    return (
        <Paper style={{ width: '100%'}}>
            <Flex direction="column">
                <Flex className={styles.header}>
                    <Flex gap="20px" align="center">
                        <Text className={styles.allText}>All ({data.meta.total})</Text>
                    </Flex>
                    <TextInput
                        placeholder="Search user"
                        value={searchTerm}
                        onChange={(event) => setSearchTerm(event.currentTarget.value)}
                        size="sm"
                        style={{ maxWidth: 500, minWidth: 300, }}
                    />

                </Flex>

                <div className={styles.tableWrapper}>
                    {data && data.meta ? (
                        <DataTable
                            fetching={isLoading}
                            borderRadius="sm"
                            withTableBorder
                            striped
                            highlightOnHover
                            classNames={{
                                root: styles.tableRoot,
                                table: styles.table,
                                header: styles.tableHeader,
                                pagination: styles.pagination,
                            }}
                            records={data.items}
                            columns={columns}
                            totalRecords={data?.meta?.total ?? 0}
                            recordsPerPage={perPage}
                            page={page}
                            onPageChange={setPage}
                            recordsPerPageOptions={[10, 20, 30, 50]}
                            onRecordsPerPageChange={setPerPage}
                            recordsPerPageLabel="Displayed on the page"
                            paginationSize="md"
                            paginationText={({ from, to }) => `${from}-${to} / ${data.meta.total}`}
                            sortStatus={{
                                columnAccessor: sortBy ?? 'name',
                                direction: sortDirection ?? 'asc',
                            }}

                            onSortStatusChange={({ columnAccessor, direction }) => {
                                setSortBy(columnAccessor);
                                setSortDirection(direction);
                            }}
                        />
                    ) : (
                        <div>Loading...</div>
                    )}
                </div>
            </Flex>

            <DeleteUserModal
                opened={deleteModalOpened}
                close={() => setDeleteModalOpened(false)}
                userid={currentUserId}
                onSuccess={fetchData}
            />

            <EditUserModal
                opened={opened}
                close={close}
                initialValue={currentUser}
                isEditing={isEditing}
                onSuccess={fetchData}
            />

            
        </Paper>

        
    )
}