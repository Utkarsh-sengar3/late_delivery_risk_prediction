import { AppShell } from "@/components/app-shell";
export function PageLayout({ title, subtitle, children, actions }: { title: string; subtitle: string; children: React.ReactNode; actions?: React.ReactNode }) {
  return <AppShell title={title} subtitle={subtitle} actions={actions}>{children}</AppShell>;
}
