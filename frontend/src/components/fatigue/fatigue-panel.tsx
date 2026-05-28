"use client";

import { useRef } from "react";
import type { FatigueMetrics, DrowsinessLevel } from "@/types";

interface FatiguePanelProps {
  metrics: FatigueMetrics | null;
  isEnabled: boolean;
}

const LEVEL_CONFIG: Record<DrowsinessLevel, { color: string; label: string; emoji: string }> = {
  alert:    { color: "var(--success)",          label: "Alert",    emoji: "✅" },
  mild:     { color: "var(--warning)",          label: "Mild",     emoji: "⚠️" },
  moderate: { color: "#f97316",                  label: "Moderate", emoji: "🟠" },
  severe:   { color: "var(--danger)",           label: "Severe",   emoji: "🔴" },
  critical: { color: "var(--fatigue-critical)", label: "Critical", emoji: "🚨" },
};

export function FatiguePanel({ metrics, isEnabled }: FatiguePanelProps) {
  const level = metrics?.drowsinessLevel ?? "alert";
  const config = LEVEL_CONFIG[level];

  return (
    <div className="panel flex flex-col h-full" style={{ minHeight: 160 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
        <p style={{ fontFamily: "var(--font-display)", fontSize: 11, letterSpacing: "0.08em", textTransform: "uppercase", color: "var(--text-tertiary)" }}>
          Eye Fatigue
        </p>
        <span style={{ fontSize: 11 }}>
          {isEnabled ? (
            <span style={{ color: "var(--success)", fontFamily: "var(--font-mono)", fontSize: 10 }}>● LIVE</span>
          ) : (
            <span style={{ color: "var(--text-tertiary)", fontFamily: "var(--font-mono)", fontSize: 10 }}>○ OFF</span>
          )}
        </span>
      </div>

      {/* Fatigue score */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 10 }}>
        <div style={{ display: "flex", alignItems: "flex-end", gap: 6 }}>
          <span style={{ fontFamily: "var(--font-display)", fontSize: "2.2rem", fontWeight: 700, color: config.color, lineHeight: 1 }}>
            {metrics ? Math.round(metrics.fatigueScore) : "--"}
          </span>
          <span style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 4 }}>/ 100</span>
        </div>

        {/* Status badge */}
        <div style={{
          display: "inline-flex",
          alignItems: "center",
          gap: 5,
          padding: "3px 8px",
          borderRadius: 99,
          background: config.color + "18",
          border: `1px solid ${config.color}44`,
          width: "fit-content",
        }}>
          <span style={{ fontSize: 10 }}>{config.emoji}</span>
          <span style={{ fontSize: 10, color: config.color, fontWeight: 600, fontFamily: "var(--font-mono)" }}>
            {config.label}
          </span>
        </div>

        {/* Sub-metrics */}
        {metrics && (
          <div style={{ display: "flex", flexDirection: "column", gap: 4, marginTop: 4 }}>
            <MetricRow label="Blink Rate" value={`${Math.round(metrics.blinkRate)}/min`} />
            <MetricRow label="EAR" value={metrics.eyeAspectRatio.toFixed(3)} />
            <MetricRow label="Confidence" value={`${Math.round(metrics.confidence * 100)}%`} />
          </div>
        )}
      </div>

      {!isEnabled && (
        <p style={{ fontSize: 11, color: "var(--text-tertiary)", marginTop: 8 }}>
          Enable camera to detect fatigue
        </p>
      )}
    </div>
  );
}

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between" }}>
      <span style={{ fontSize: 10, color: "var(--text-tertiary)" }}>{label}</span>
      <span style={{ fontSize: 10, color: "var(--text-secondary)", fontFamily: "var(--font-mono)" }}>{value}</span>
    </div>
  );
}
