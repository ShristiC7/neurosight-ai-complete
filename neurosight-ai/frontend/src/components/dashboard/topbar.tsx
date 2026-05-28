"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useWebSocket } from "@/hooks/use-websocket";
import { useDashboardStore } from "@/store/dashboard-store";

export function TopBar() {
  const { isConnected, isReconnecting } = useWebSocket();
  const { alertCount, clearAlerts, dashboard } = useDashboardStore();

  return (
    <header
      style={{
        height: "var(--header-height)",
        borderBottom: "1px solid var(--border-subtle)",
        display: "flex",
        alignItems: "center",
        paddingLeft: 24,
        paddingRight: 24,
        gap: 16,
        background: "rgba(10,15,30,0.8)",
        backdropFilter: "blur(12px)",
        zIndex: 10,
        flexShrink: 0,
      }}
    >
      {/* Page title */}
      <div style={{ flex: 1 }}>
        <h1 style={{
          fontFamily: "var(--font-display)",
          fontSize: 16,
          fontWeight: 700,
          color: "var(--text-primary)",
          letterSpacing: "0.02em",
        }}>
          Cognitive Dashboard
        </h1>
        <p style={{ fontSize: 11, color: "var(--text-tertiary)", marginTop: 1 }}>
          Real-time multimodal monitoring
        </p>
      </div>

      {/* Session indicator */}
      {dashboard.isMonitoring && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            padding: "4px 12px",
            borderRadius: 99,
            background: "rgba(16,185,129,0.08)",
            border: "1px solid rgba(16,185,129,0.2)",
          }}
        >
          <span className="status-dot active" />
          <span style={{ fontSize: 11, fontWeight: 600, color: "var(--success)", fontFamily: "var(--font-mono)" }}>
            MONITORING ACTIVE
          </span>
        </motion.div>
      )}

      {/* WS connection pill */}
      <div style={{
        display: "flex",
        alignItems: "center",
        gap: 6,
        padding: "4px 10px",
        borderRadius: 99,
        background: isConnected ? "rgba(16,185,129,0.06)" : "rgba(239,68,68,0.06)",
        border: `1px solid ${isConnected ? "rgba(16,185,129,0.15)" : "rgba(239,68,68,0.15)"}`,
        fontSize: 10,
        fontFamily: "var(--font-mono)",
        color: isConnected ? "var(--success)" : isReconnecting ? "var(--warning)" : "var(--danger)",
      }}>
        <span style={{
          width: 6,
          height: 6,
          borderRadius: "50%",
          background: "currentColor",
          flexShrink: 0,
          ...(isConnected ? { boxShadow: "0 0 6px currentColor" } : {}),
        }} />
        {isReconnecting ? "RECONNECTING" : isConnected ? "LIVE" : "OFFLINE"}
      </div>

      {/* Alert bell */}
      {alertCount > 0 && (
        <motion.button
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          onClick={clearAlerts}
          style={{
            position: "relative",
            width: 36,
            height: 36,
            borderRadius: "50%",
            background: "rgba(239,68,68,0.1)",
            border: "1px solid rgba(239,68,68,0.3)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            cursor: "pointer",
            color: "#ef4444",
          }}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
            <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
            <path d="M13.73 21a2 2 0 0 1-3.46 0" />
          </svg>
          <span style={{
            position: "absolute",
            top: -4,
            right: -4,
            width: 16,
            height: 16,
            borderRadius: "50%",
            background: "#ef4444",
            color: "white",
            fontSize: 9,
            fontWeight: 700,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}>
            {alertCount > 9 ? "9+" : alertCount}
          </span>
        </motion.button>
      )}
    </header>
  );
}
