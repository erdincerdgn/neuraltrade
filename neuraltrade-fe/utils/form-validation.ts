export const isValidTurkishPhoneNumber = (phone: string): boolean => {
    const turkishPhoneRegex = /^(\+90|0)?[0-9]{10}$/;
    return turkishPhoneRegex.test(phone.replace(/[\s-]/g, ''));
};

// English phone validation will be added later

export const isValidUrl = (url: string): boolean => {
    const urlRegex = /^(https?:\/\/)?([\da-z.-]+)\.([a-z.]{2,6})([/\w .-]*)*\/?$/;
    return urlRegex.test(url);
}

