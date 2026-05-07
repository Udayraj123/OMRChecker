"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";
import {
  Bell,
  BrainCircuit,
  ChevronLeft,
  LogOut,
  Menu,
  Moon,
  Search,
  ShieldCheck,
  Sparkles,
  Sun,
  X,
} from "lucide-react";
import { useEffect, useState } from "react";
import { navItems } from "./data";

type ShellProps = {
  children: React.ReactNode;
  title: string;
  eyebrow?: string;
};

export function AppShell({ children, title, eyebrow = "MinSU AI OMR" }: ShellProps) {
  const pathname = usePathname();
  const router = useRouter();
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [dark, setDark] = useState(false);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);

  async function handleLogout() {
    await fetch("/api/logout", { method: "POST" });
    router.replace("/login");
    router.refresh();
  }

  const sidebar = (
    <aside
      className={`glass-panel flex h-full flex-col rounded-none border-y-0 border-l-0 transition-all duration-300 ${
        collapsed ? "w-[86px]" : "w-[276px]"
      }`}
    >
      <div className="flex h-20 items-center gap-3 px-5">
        <div className="flex h-11 w-11 items-center justify-center rounded-lg bg-emerald-600 text-white shadow-lg shadow-emerald-700/20">
          <BrainCircuit size={24} />
        </div>
        {!collapsed && (
          <div>
            <p className="text-lg font-semibold text-slate-950 dark:text-white">MinSU ScanAI</p>
            <p className="text-xs text-slate-500 dark:text-slate-400">Entrance exam scanner</p>
          </div>
        )}
      </div>

      <nav className="flex-1 space-y-1 px-3 py-4">
        {navItems.map((item) => {
          const active = pathname === item.href;
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              onClick={() => setMobileOpen(false)}
              className={`group flex items-center gap-3 rounded-lg px-3 py-3 text-sm font-medium transition ${
                active
                  ? "bg-slate-900 text-white shadow-lg shadow-slate-900/15 dark:bg-sky-300 dark:text-slate-950"
                  : "text-slate-600 hover:bg-slate-100 hover:text-slate-950 dark:text-slate-300 dark:hover:bg-white/10 dark:hover:text-white"
              }`}
              title={collapsed ? item.label : undefined}
            >
              <Icon size={20} />
              {!collapsed && <span>{item.label}</span>}
            </Link>
          );
        })}
      </nav>

      <div className="p-3">
        {!collapsed && (
          <div className="mb-3 rounded-lg border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-950 dark:border-emerald-400/20 dark:bg-emerald-400/10 dark:text-emerald-100">
            <div className="mb-2 flex items-center gap-2 font-semibold">
              <Sparkles size={16} />
              AI scan engine
            </div>
            <p className="text-xs leading-5 opacity-80">OMRChecker backend ready for batch scanning.</p>
          </div>
        )}
        <button
          onClick={() => setCollapsed((value) => !value)}
          className="flex w-full items-center justify-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600 transition hover:border-slate-300 hover:text-slate-950 dark:border-white/10 dark:bg-white/5 dark:text-slate-300"
        >
          <ChevronLeft className={`transition ${collapsed ? "rotate-180" : ""}`} size={18} />
          {!collapsed && "Collapse"}
        </button>
      </div>
    </aside>
  );

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,rgba(37,99,235,0.12),transparent_30%),linear-gradient(180deg,#f6fbff_0%,#eef5f1_100%)] text-slate-950 dark:bg-[radial-gradient(circle_at_top_left,rgba(56,189,248,0.14),transparent_32%),linear-gradient(180deg,#07111f_0%,#0a1726_100%)] dark:text-white">
      <div className="flex min-h-screen">
        <div className="hidden lg:block">{sidebar}</div>

        <AnimatePresence>
          {mobileOpen && (
            <motion.div
              className="fixed inset-0 z-50 bg-slate-950/40 backdrop-blur-sm lg:hidden"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <motion.div
                className="h-full w-[286px]"
                initial={{ x: -300 }}
                animate={{ x: 0 }}
                exit={{ x: -300 }}
              >
                {sidebar}
              </motion.div>
              <button
                className="absolute right-4 top-4 rounded-lg bg-white p-2 text-slate-900"
                onClick={() => setMobileOpen(false)}
                aria-label="Close navigation"
              >
                <X size={20} />
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        <main className="min-w-0 flex-1">
          <header className="sticky top-0 z-30 border-b border-slate-200/70 bg-white/80 px-4 py-3 backdrop-blur-xl dark:border-white/10 dark:bg-slate-950/65 sm:px-6">
            <div className="flex items-center gap-3">
              <button
                className="rounded-lg border border-slate-200 bg-white p-2 lg:hidden dark:border-white/10 dark:bg-white/5"
                onClick={() => setMobileOpen(true)}
                aria-label="Open navigation"
              >
                <Menu size={20} />
              </button>
              <div className="min-w-0 flex-1">
                <p className="text-xs font-semibold uppercase tracking-[0.22em] text-emerald-700 dark:text-emerald-300">
                  {eyebrow}
                </p>
                <h1 className="truncate text-xl font-semibold sm:text-2xl">{title}</h1>
              </div>
              <div className="hidden min-w-[280px] items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-500 dark:border-white/10 dark:bg-white/5 md:flex">
                <Search size={17} />
                Search scans, LRN, barcode
              </div>
              <button
                className="rounded-lg border border-slate-200 bg-white p-2 transition hover:bg-slate-50 dark:border-white/10 dark:bg-white/5 dark:hover:bg-white/10"
                onClick={() => setDark((value) => !value)}
                aria-label="Toggle theme"
              >
                {dark ? <Sun size={19} /> : <Moon size={19} />}
              </button>
              <button className="hidden rounded-lg border border-slate-200 bg-white p-2 transition hover:bg-slate-50 dark:border-white/10 dark:bg-white/5 sm:block">
                <Bell size={19} />
              </button>
              <div className="hidden items-center gap-2 rounded-lg bg-slate-900 px-3 py-2 text-sm font-semibold text-white dark:bg-sky-300 dark:text-slate-950 sm:flex">
                <ShieldCheck size={17} />
                Registrar
              </div>
              <button
                className="rounded-lg border border-slate-200 bg-white p-2 transition hover:bg-slate-50 dark:border-white/10 dark:bg-white/5 dark:hover:bg-white/10"
                onClick={handleLogout}
                aria-label="Logout"
                title="Logout"
              >
                <LogOut size={19} />
              </button>
            </div>
          </header>

          <motion.div
            key={pathname}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.28 }}
            className="p-4 sm:p-6 lg:p-8"
          >
            {children}
          </motion.div>
        </main>
      </div>
    </div>
  );
}

export function Card({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return <section className={`glass-panel rounded-lg ${className}`}>{children}</section>;
}

export function StatusBadge({ status }: { status: string }) {
  const tone =
    status === "Passed" || status === "Validated" || status === "Completed" || status === "Exported" || status === "Ready"
      ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-400/15 dark:text-emerald-200"
      : status === "Review" || status === "Scanning"
        ? "bg-amber-100 text-amber-800 dark:bg-amber-400/15 dark:text-amber-200"
        : "bg-rose-100 text-rose-800 dark:bg-rose-400/15 dark:text-rose-200";
  return <span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${tone}`}>{status}</span>;
}
