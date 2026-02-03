// import { ListingStatus } from '@/utils/enums/listing';

export const getNavbarList = () => [
  {
    path: '/dashboard',
    label: 'Dashboard',
  },
  {
    path: '/listing',
    label: 'Job',
    isNested: true,
    subItems: [
      { label: 'Draft', path: `/dashboard` },
      { label: 'Pending Approval', path: `/listing?status=$/dashboard` },
      { label: 'Revision', path: `/dashboard` },
      { label: 'Active', path: `/dashboard` },
      { label: 'Inactive', path: `/dashboard` },
      { label: 'Sold', path: `/dashboard` },
    ],
  },
  {
    path: '/user',
    label: 'User',
  },
  {
    path: '/post',
    label: 'Post'
  },
  {
    path: '/article',
    label: 'Article'
  },
  {
    path: '/comment',
    label: 'Comment'
  },
  {
    path: '/complaint',
    label: 'Complaints'
  },
  {
    path: '/feedback',
    label: 'Feedback',
  },
  {
    path: '/settings',
    label: 'Setting',
  },
  {
    key: 'logout',
    label: 'Log Out',
  },
];
