export const formatters = {
    city: (text: string) =>
        text
            .toLowerCase()
            .replace(/^\w/, (c) => c.toUpperCase())
            .replace(/\((.*?)\)/gi, (_, m) => `(${m.toLowerCase()})`),
    district: (text: string) => text.toLowerCase().replace(/^\w/, (c) => c.toUpperCase()),
    neighborhood: (text: string) =>
        text
            .toLowerCase()
            .replace(/^\w/, (c) => c.toUpperCase())
            .replace('mah.', 'Mah.'),
    street: (text: string) =>
        text
            .toLowerCase()
            .replace(/\b(sokak|sk|cadde|cd|bulvar|blv|mahallesi|mah)\b/gi, (word) => word.toLowerCase())
            .replace(/\b\w+\b/g, (word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()),
};

// export const formatAddressForBreadcrumb = (listing?: ListingDetail) => {
//     if (!listing) return [];

//     return ['city', 'district', 'neighborhood', 'street'].map((key) =>  listing[key as keyof typeof formatters]
//         ? formatters[key as keyof typeof formatters](listing[key as keyof typeof formatters])
//         : ''
//     );
// };