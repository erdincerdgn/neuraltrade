// import { DateFilter } from "@/enum/listing";
// import { createFormContext } from "@mantine/form";
// import { useCallback } from "react";

// export interface SearchFormValues {
  
//   dateFilter?: DateFilter;

  
//   city?: string[];
//   district?: string[];

//   // Modal States
//   saveModalOpened: boolean;
//   allFilterDrawerOpened: boolean;

//   lastUpdatedField?: string;
// }

// export const [SearchFormProvider, useSearchFormContext, useSearchForm] =
//   createFormContext<SearchFormValues>();

// export const useOptimizedSearchForm = () => {
//   const form = useSearchFormContext();

//   const setFieldValue = useCallback(
//     (field: keyof SearchFormValues, value: any) => {
//       form.setValues((current) => ({
//         ...current,
//         [field]: value,
//         lastUpdatedField: field,
//       }));
//     },
//     [form]
//   );

//   return {
//     ...form,
//     setFieldValue,
//   };
// };