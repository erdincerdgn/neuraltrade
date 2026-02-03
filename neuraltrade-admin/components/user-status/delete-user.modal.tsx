import { Flex, Modal, Button , Text } from '@mantine/core';
import React from 'react';
import { notificationError, notificationSuccess } from '@/utils/notification-manager';

interface IModal {
  opened: boolean;
  close: () => void;
  userid: string;
  onSuccess: () => void;
}

function DeleteUserModel({ opened, close, userid , onSuccess }: IModal) {
  const deleteUser = async (id: string) => {
    try {
      // Delete
      notificationSuccess({ message: 'User successfully deleted' });
      onSuccess();
    } catch (error) {
      notificationError({ message: 'An error occurred while deleting the user' });
    }
    close();
  };

  return (
    <Modal
      opened={opened}
      onClose={close}
      title="Delete User"
      transitionProps={{ transition: 'fade', duration: 400, timingFunction: 'linear' }}
    >
      <Flex gap={10} mt={20}>
          <Text>Are you sure you want to delete this User?</Text>
        <Button w={100} onClick={() => console.log('...')}>Yes</Button>
        <Button w={100} variant="outline" onClick={close}>
          No
        </Button>
      </Flex>
    </Modal>
  );
}

export default DeleteUserModel;

// deleteUser(userid) to 33th line