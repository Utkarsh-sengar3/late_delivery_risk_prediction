import { ArrowDownRight, ArrowUpRight, Minus, type LucideIcon } from "lucide-react";

export function SectionHeader({ title, description, action }: { title: string; description?: string; action?: React.ReactNode }) {
  return <div className="section-header"><div><h2>{title}</h2>{description && <p>{description}</p>}</div>{action && <div>{action}</div>}</div>;
}

export function MetricCard({ label, value, note, icon: Icon, tone = "blue" }: {
  label: string; value: string; note: string; icon: LucideIcon; tone?: "blue"|"teal"|"amber"|"red"|"violet";
}) {
  return <div className="metric-card">
    <div className="metric-top"><span>{label}</span><span className={`metric-icon ${tone}`}><Icon size={17}/></span></div>
    <div className="metric-value">{value}</div>
    <div className="metric-note"><span className="metric-note-dot" />{note}</div>
  </div>;
}

export function Panel({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <section className={`panel ${className}`}>{children}</section>;
}

export function StatusPill({ children, tone = "neutral" }: { children: React.ReactNode; tone?: "good"|"warning"|"danger"|"neutral"|"info" }) {
  return <span className={`status-pill ${tone}`}><span />{children}</span>;
}

export function EmptyState({ title, description, action }: { title: string; description: string; action?: React.ReactNode }) {
  return <div className="empty-state"><div className="empty-state-mark"><Minus size={20}/></div><h3>{title}</h3><p>{description}</p>{action}</div>;
}
