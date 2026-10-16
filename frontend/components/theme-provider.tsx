 "use client";

import { createContext, useContext, useEffect, useMemo, useState } from "react";

type Theme = "light" | "dark";
type Accent = "blue" | "teal" | "violet" | "graphite";
type Preferences = { theme: Theme; accent: Accent; compact: boolean };
type ContextValue = Preferences & { setTheme: (v: Theme) => void; setAccent: (v: Accent) => void; setCompact: (v: boolean) => void };
const ThemeContext = createContext<ContextValue | null>(null);

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setThemeState] = useState<Theme>("light");
  const [accent, setAccentState] = useState<Accent>("blue");
  const [compact, setCompactState] = useState(false);

  useEffect(() => {
    const saved = localStorage.getItem("ldr-preferences");
    if (saved) {
      try {
        const p = JSON.parse(saved) as Preferences;
        setThemeState(p.theme || "light");
        setAccentState(p.accent || "blue");
        setCompactState(Boolean(p.compact));
      } catch {}
    }
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.documentElement.dataset.accent = accent;
    document.documentElement.dataset.density = compact ? "compact" : "comfortable";
    localStorage.setItem("ldr-preferences", JSON.stringify({ theme, accent, compact }));
  }, [theme, accent, compact]);

  const value = useMemo(() => ({
    theme, accent, compact,
    setTheme: setThemeState, setAccent: setAccentState, setCompact: setCompactState,
  }), [theme, accent, compact]);

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  const value = useContext(ThemeContext);
  if (!value) throw new Error("useTheme must be used within ThemeProvider");
  return value;
}
