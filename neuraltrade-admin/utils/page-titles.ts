interface PageTitle {
  path: string;
  title: string;
}

const pageTitles: PageTitle[] = [
  { path: '/dashboard', title: 'Admin Dashboard' },
  { path: '/listings/drafts', title: 'Taslak İlanlar' },
  { path: '/listings/pending', title: 'Onay Bekleyen İlanlar' },
  { path: '/listings/revisions', title: 'Revizyon Bekleyen İlanlar' },
  { path: '/listings/published', title: 'Yayındaki İlanlar' },
  { path: '/inquiries', title: 'Gelen Talepler' },
  { path: '/offers', title: 'Teklifler' },
  { path: '/users', title: 'Kullanıcılar' },
  { path: '/real-estate/offices', title: 'Emlak Ofisleri' },
  { path: '/real-estate/employees', title: 'Emlak Çalışanları' },
  { path: '/settings', title: 'Ayarlar' },
];

export const getPageTitle = (path: string): string => {
  const page = pageTitles.find((p) => p.path === path);
  return page?.title || 'NeuralTrade Dashboard';
};
