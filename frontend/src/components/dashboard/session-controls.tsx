"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { useDashboardStore } from "@/store/dashboard-store";
import { apiClient } from "@/lib/api-client";
import { useAuthStore } from "@/store/auth-store";
import type { WorkSession } from "@/types";

export function SessionControls() {
  const { user } = useAuthStore();
  const {
    isMonitoringActive,
    isCameraEnabled,
    isMicEnabled,
    isKeyboardTracking,
    dashboard,
    toggleMonitoring,
    toggleCamera,
    toggleMic,
    toggleKeyboardTracking,
    startSession,
    endSession,
  } = useDashboardStore();

  const [isStarting, setIsStarting] = useState(false);

  const handleToggleSession = async () => {
    if (isMonitoringActive) {
      // End session
      if (dashboard.activeSession) {
        try {
          await apiClient.post(`/sessions/${dashboard.activeSession.id}/end`);
        } catch {}
      }
      endSession();
      toggleMonitoring();
    } else {
      // Start session
      setIsStarting(true);
      try {
        const session = await apiClient.post<WorkSession>("/sessions/start");
        startSession(session);
        toggleMonitoring();
      } catch (err) {
        console.error("Failed to start session:", err);
      } finally {
        setIsStarting(false);
      }
    }
  };

  const sensors = [
    {
      id: "camera",
      label: "Camera",
      icon: "📸",
      enabled: isCameraEnabled,
      toggle: toggleCamera,
      description: "Fatigue detection",
    },
    {
      id: "mic",
      label: "Mic",
      icon: "🎙️",
      enabled: isMicEnabled,
      toggle: toggleMic,
      description: "Stress analysis",
    },
    {
      id: "keyboard",
      label: "Keys",
      icon: "⌨️",
      enabled: isKeyboardTracking,
      toggle: toggleKeyboardTracking,
      description: "Behavioral",
    },
  ];

  return (
    <div className="panel flex flex-col h-full" style={{ minHeight: 160 }}>
      <p
        style={{
          fontFamily: "var(--font-display)",
          fontSize: 11,
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          color: "var(--text-tertiary)",
          marginBottom: 12,
        }}
      >
        Monitoring Controls
      </p>

      {/* Main toggle */}
      <motion.button
        onClick={handleToggleSession}
        disabled={isStarting}
        whileTap={{ scale: 0.97 }}
        style={{
          width: "100%",
          padding: "10px 0",
          borderRadius: 10,
          border: "none",
          cursor: isStarting ? "not-allowed" : "pointer",
          fontFamily: "var(--font-display)",
          fontWeight: 700,
          fontSize: 13,
          letterSpacing: "0.04em",
          transition: "all 200ms",
          background: isMonitoringActive
            ? "rgba(239,68,68,0.15)"
            : "linear-gradient(135deg, #4f6ef7, #00d9c8)",
          color: isMonitoringActive ? "#ef4444" : "white",
          border: isMonitoringActive ? "1px solid rgba(239,68,68,0.3)" : "none",
          boxShadow: isMonitoringActive ? "none" : "0 0 20px rgba(79,110,247,0.3)",
          marginBottom: 12,
        }}
      >
        {isStarting ? "Starting..." : isMonitoringActive ? "⬛ Stop Session" : "▶ Start Monitoring"}
      </motion.button>

      {/* Sensor toggles */}
      <div style={{ display: "flex", gap: 8 }}>
        {sensors.map((sensor) => (
          <button
            key={sensor.id}
            onClick={sensor.toggle}
            title={sensor.description}
            style={{
              flex: 1,
              padding: "8px 4px",
              borderRadius: 8,
              border: `1px solid ${sensor.enabled ? "var(--border-strong)" : "var(--border-subtle)"}`,
              background: sensor.enabled ? "rgba(79,110,247,0.08)" : "transparent",
              cursor: "pointer",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 4,
              transition: "all 150ms",
            }}
          >
            <span style={{ fontSize: 16, opacity: sensor.enabled ? 1 : 0.3 }}>{sensor.icon}</span>
            <span
              style={{
                fontSize: 9,
                fontFamily: "var(--font-mono)",
                color: sensor.enabled ? "var(--accent-primary)" : "var(--text-tertiary)",
                letterSpacing: "0.05em",
              }}
            >
              {sensor.label}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
