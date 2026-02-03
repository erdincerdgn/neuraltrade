export const formatPhoneNumber = (phoneNumber: string) => {
    if (phoneNumber && phoneNumber.length === 11 && phoneNumber.startsWith('0')) {
      return `${phoneNumber.substring(0, 1)} (${phoneNumber.substring(1, 4)}) ${phoneNumber.substring(4, 7)} ${phoneNumber.substring(7, 9)} ${phoneNumber.substring(9, 11)}`;
    }
    if (phoneNumber && phoneNumber.length === 10) {
      return `0 (${phoneNumber.substring(0, 3)}) ${phoneNumber.substring(3, 6)} ${phoneNumber.substring(6, 8)} ${phoneNumber.substring(8, 10)}`;
    }
    return phoneNumber;
  };