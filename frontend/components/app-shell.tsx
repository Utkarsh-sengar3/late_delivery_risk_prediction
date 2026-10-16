 "use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import {
  Activity, ArrowUpRight, Bell, ChevronDown, ChevronLeft, ChevronRight, CircleHelp,
  Command, Gauge, History, Info, Layers3, LayoutDashboard, Menu, Moon, Search,
  Settings2, ShieldCheck, Sparkles, Sun, Target, UserRound, X, ChartNoAxesCombined,
} from "lucide-react";
import { navGroups } from "@/lib/data";
import { useTheme } from "@/components/theme-provider";

const iconMap: Record<string, React.ElementType> = {
  layout: LayoutDashboard, target: Target, chart: ChartNoAxesCombined, layers: Layers3,
  activity: Activity, sparkles: Sparkles, history: History, info: Info,
  user: UserRound, settings: Settings2,
};

export function AppShell({ children, title, subtitle, actions }: {
  children: React.ReactNode; title: string; subtitle: string; actions?: React.ReactNode;
}) {
  const pathname = usePathname();
  const { theme, setTheme } = useTheme();
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);

  const sidebar = (
    <div className={`sidebar-inner ${collapsed ? "is-collapsed" : ""}`}>
      <Link href="/dashboard" className="brand">
        <span className="brand-mark"><ShieldCheck size={20} strokeWidth={2.2} /></span>
        {!collapsed && <span className="brand-copy"><strong>Northstar</strong><small>SUPPLY CHAIN INTELLIGENCE</small></span>}
      </Link>
      <div className="workspace-select">
        <span className="workspace-avatar">LR</span>
        {!collapsed && <span className="workspace-copy"><strong>Late Delivery Risk</strong><small>Model workspace</small></span>}
        {!collapsed && <ChevronDown size={15} className="muted-icon" />}
      </div>
      <nav className="nav-groups">
        {navGroups.map(group => (
          <div className="nav-group" key={group.title}>
            {!collapsed && <div className="nav-heading">{group.title}</div>}
            {group.items.map(item => {
              const Icon = iconMap[item.icon] || Gauge;
              const active = pathname === item.href;
              return <Link onClick={() => setMobileOpen(false)} key={item.href} href={item.href}
                title={collapsed ? item.label : undefined} className={`nav-item ${active ? "active" : ""}`}>
                <Icon size={18} strokeWidth={active ? 2.2 : 1.8} />
                {!collapsed && <span>{item.label}</span>}
                {!collapsed && active && <span className="nav-active-dot" />}
              </Link>;
            })}
          </div>
        ))}
      </nav>
      {!collapsed && <div className="sidebar-bottom">
        <div className="help-card">
          <div className="help-icon"><CircleHelp size={17} /></div>
          <strong>Need a hand?</strong>
          <p>Explore the model notes and project methodology.</p>
          <Link href="/about">View documentation <ArrowUpRight size={13} /></Link>
        </div>
        <div className="sidebar-user">
          <div className="avatar">US</div>
          <div className="user-copy"><strong>Utkarsh Sengar</strong><small>Project developer</small></div>
          <Link href="/developer" aria-label="Developer profile"><ChevronRight size={16} /></Link>
        </div>
      </div>}
    </div>
  );

  return (
    <div className={`app-frame ${collapsed ? "sidebar-collapsed" : ""}`}>
      <aside className="desktop-sidebar">{sidebar}<button className="collapse-button" onClick={() => setCollapsed(!collapsed)} aria-label="Toggle sidebar">{collapsed ? <ChevronRight size={16}/> : <ChevronLeft size={16}/>}</button></aside>
      {mobileOpen && <div className="mobile-backdrop" onClick={() => setMobileOpen(false)}><aside className="mobile-sidebar" onClick={e => e.stopPropagation()}>{sidebar}</aside></div>}
      <main className="main-area">
        <header className="topbar">
          <div className="topbar-left">
            <button className="icon-button mobile-menu-button" onClick={() => setMobileOpen(true)} aria-label="Open navigation"><Menu size={19}/></button>
            <div className="breadcrumbs"><span>Workspace</span><ChevronRight size={13}/><strong>{title}</strong></div>
          </div>
          <div className="topbar-actions">
            <button className="search-trigger" onClick={() => setSearchOpen(!searchOpen)}><Search size={16}/><span>Search</span><kbd>⌘ K</kbd></button>
            <button className="icon-button" onClick={() => setTheme(theme === "light" ? "dark" : "light")} title="Toggle appearance" aria-label="Toggle appearance">{theme === "light" ? <Moon size={17}/> : <Sun size={17}/>}</button>
            <Link className="icon-button notification-button" href="/settings" aria-label="Settings"><Bell size={17}/><i /></Link>
            <Link href="/developer" className="top-avatar" aria-label="Developer profile">US</Link>
          </div>
        </header>
        {searchOpen && <div className="search-overlay">
          <div className="search-panel"><div className="search-panel-head"><Search size={18}/><input autoFocus placeholder="Search pages…" onKeyDown={e => { if(e.key==="Escape") setSearchOpen(false); }}/><kbd>ESC</kbd><button className="icon-button" onClick={() => setSearchOpen(false)}><X size={16}/></button></div>
            <div className="search-results">{navGroups.flatMap(g=>g.items).map(item=><Link key={item.href} href={item.href} onClick={()=>setSearchOpen(false)}>{item.label}<ArrowUpRight size={14}/></Link>)}</div>
          </div>
        </div>}
        <div className="page-content">
          <div className="page-heading">
            <div><div className="eyebrow"><span className="eyebrow-dot" /> RISK INTELLIGENCE PLATFORM</div><h1>{title}</h1><p>{subtitle}</p></div>
            {actions && <div className="heading-actions">{actions}</div>}
          </div>
          {children}
          <footer className="app-footer"><span>Northstar <span className="footer-sep">/</span> Late Delivery Risk</span><span>Model evaluation · Phase 5 &amp; 6 artifacts</span></footer>
        </div>
      </main>
    </div>
  );
}
