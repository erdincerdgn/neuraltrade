import { create } from 'zustand';
import { devtools } from 'zustand/middleware';

// export type StoreInterface = ListingSliceInterface;

// export const useStore = create<StoreInterface>()(
//   devtools(
//     (...a) => ({
//       // ...createListingSlice(...a),
//     }),
//     { name: 'store' }
//   )
// );