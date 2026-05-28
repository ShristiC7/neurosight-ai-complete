"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";
import { useDashboardStore } from "@/store/dashboard-store";
import { useWebSocket } from "@/hooks/use-websocket";

const NAV_ITEMS = [
  { href: "/dashboard",  label: "Dashboard",   icon: "⬡" },
  { href: "/analytics",  label: "Analytics",   icon: "◈" },
  { href: "/sessions",   label: "Sessions",    icon: "◫" },
  { href: "/insights",   label: "AI Insights", icon: "◉" },
  { href: "/settings",   label: "Settings",    icon: "◌" },
];

export function Sidebar() {
  const pathname = usePathname();
  const { isMonitoringActive, dashboard } = useDashboardStore();
  const { isConnected } = useWebSocket();

  return (
    <aside
      style={{
        width: "var(--sidebar-width)",
        flexShrink: 0,
        background: "var(--bg-surface)",
        borderRight: "1px solid var(--border-subtle)",
        display: "flex",
        flexDirection: "column",
        padding: "0",
        zIndex: 20,
      }}
    >
      {/* Logo */}
      <div
        style={{
          height: "var(--header-height)",
          padding: "0 20px",
          display: "flex",
          alignItems: "center",
          gap: 10,
          borderBottom: "1px solid var(--border-subtle)",
        }}
      >
        <div
          style={{
            width: 30,
            height: 30,
            borderRadius: 8,
            background: "linear-gradient(135deg, #4f6ef7, #00d9c8)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 14,
            boxShadow: "0 0 16px rgba(79,110,247,0.4)",
          }}
        >
          ⬡
        </div>
        <div>
          <p style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: 13, lineHeight: 1 }}>
            NeuroSight
          </p>
          <p style={{ fontSize: 9, color: "var(--text-tertiary)", letterSpacing: "0.08em" }}>
            AI COGNITIVE
          </p>
        </div>
      </div>

      {/* Nav */}
      <nav style={{ flex: 1, padding: "16px 12px", display: "flex", flexDirection: "column", gap: 2 }}>
        {NAV_ITEMS.map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link key={item.href} href={item.href} style={{ textDecoration: "none" }}>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                  padding: "9px 12px",
                  borderRadius: 8,
                  position: "relative",
                  background: isActive ? "rgba(79,110,247,0.12)" : "transparent",
                  border: isActive ? "1px solid rgba(79,110,247,0.2)" : "1px solid transparent",
                  transition: "all 150ms",
                  cursor: "pointer",
                }}
                onMouseEnter={(e) => {
                  if (!isActive) {
                    (e.currentTarget as HTMLDivElement).style.background = "rgba(255,255,255,0.04)";
                  }
                }}
                onMouseLeave={(e) => {
                  if (!isActive) {
                    (e.currentTarget as HTMLDivElement).style.background = "transparent";
                  }
                }}
              >
                <span style={{ fontSize: 16, color: isActive ? "var(--accent-primary)" : "var(--text-tertiary)" }}>
                  {item.icon}
                </span>
                <span
                  style={{
                    fontSize: 13,
                    fontWeight: isActive ? 600 : 400,
                    color: isActive ? "var(--text-primary)" : "var(--text-secondary)",
                    fontFamily: "var(--font-display)",
                  }}
                >
                  {item.label}
                </span>
                {isActive && (
                  <motion.div
                    layoutId="sidebar-active"
                    style={{
                      position: "absolute",
                      right: 0,
                      top: "50%",
                      transform: "translateY(-50%)",
                      width: 3,
                      height: 20,
                      borderRadius: 99,
                      background: "var(--accent-primary)",
                    }}
                    transition={{ type: "spring", stiffness: 400, damping: 30 }}
                  />
                )}
              </div>
            </Link>
          );
        })}
      </nav>

      {/* Status footer */}
      <div
        style={{
          padding: "16px",
          borderTop: "1px solid var(--border-subtle)",
          display: "flex",
          flexDirection: "column",
          gap: 10,
        }}
      >
        {/* Monitoring status */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "8px 12px",
            borderRadius: 8,
            background: isMonitoringActive ? "var(--success-muted)" : "var(--bg-overlay)",
            border: `1px solid ${isMonitoringActive ? "rgba(16,185,129,0.2)" : "var(--border-subtle)"}`,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span className={`status-dot ${isMonitoringActive ? "active" : ""}`}
              style={{ background: isMonitoringActive ? "var(--success)" : "var(--text-tertiary)" }}
            />
            <span style={{ fontSize: 11, color: isMonitoringActive ? "var(--success)" : "var(--text-tertiary)", fontFamily: "var(--font-mono)" }}>
              {isMonitoringActive ? "MONITORING" : "STANDBY"}
            </span>
          </div>
        </div>

        {/* WS connection */}
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <div
            style={{
              width: 6,
              height: 6,
              borderRadius: "50%",
              background: isConnected ? "var(--success)" : "var(--danger)",
            }}
          />
          <span style={{ fontSize: 10, color: "var(--text-tertiary)", fontFamily: "var(--font-mono)" }}>
            {isConnected ? "STREAM CONNECTED" : "RECONNECTING..."}
          </span>
        </div>
      </div>
    </aside>
  );
}
