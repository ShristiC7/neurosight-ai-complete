"use client";

import { motion } from "framer-motion";
import type { BehavioralMetrics } from "@/types";

interface BehavioralPanelProps {
  metrics: BehavioralMetrics | null;
  isEnabled: boolean;
}

export function BehavioralPanel({ metrics, isEnabled }: BehavioralPanelProps) {
  const anomalyScore = metrics?.anomalyScore ?? 0;
  const isAnomalous = anomalyScore > 0.5;

  return (
    <div className="panel flex flex-col gap-3 h-full" style={{ minHeight: 200 }}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span style={{ fontSize: 16 }}>⌨</span>
          <span style={{
            fontFamily: "var(--font-display)",
            fontSize: 12,
            fontWeight: 600,
            letterSpacing: "0.05em",
            textTransform: "uppercase",
            color: "var(--text-primary)",
          }}>
            Behavioral
          </span>
        </div>
        {isEnabled && <span className="status-dot active" />}
      </div>

      {/* Anomaly indicator */}
      <div style={{
        display: "flex",
        alignItems: "center",
        gap: 10,
        padding: "8px 12px",
        borderRadius: 9,
        background: isAnomalous ? "rgba(239,68,68,0.08)" : "rgba(16,185,129,0.06)",
        border: `1px solid ${isAnomalous ? "rgba(239,68,68,0.2)" : "rgba(16,185,129,0.15)"}`,
      }}>
        <span style={{ fontSize: 18 }}>{isAnomalous ? "⚠️" : "✅"}</span>
        <div>
          <p style={{
            fontSize: 12,
            fontWeight: 700,
            color: isAnomalous ? "#ef4444" : "#10b981",
            fontFamily: "var(--font-display)",
          }}>
            {isAnomalous ? "Anomaly Detected" : "Normal Behavior"}
          </p>
          <p style={{ fontSize: 9, color: "var(--text-tertiary)" }}>
            Score: {(anomalyScore * 100).toFixed(0)}% deviation
          </p>
        </div>
      </div>

      {/* Metrics */}
      {metrics && (
        <div className="grid grid-cols-2 gap-2">
          <BehaviorMetricTile
            label="Typing Speed"
            value={`${Math.round(metrics.typingSpeed)} WPM`}
            status={metrics.typingSpeed > 30 ? "normal" : "low"}
          />
          <BehaviorMetricTile
            label="Error Rate"
            value={`${metrics.errorRate.toFixed(1)}/min`}
            status={metrics.errorRate < 5 ? "normal" : "high"}
          />
          <BehaviorMetricTile
            label="Mouse Entropy"
            value={metrics.mouseMovementEntropy.toFixed(2)}
            status={metrics.mouseMovementEntropy > 0.3 ? "normal" : "low"}
          />
          <BehaviorMetricTile
            label="App Switches"
            value={`${Math.round(metrics.appSwitchFrequency)}/hr`}
            status={metrics.appSwitchFrequency < 20 ? "normal" : "high"}
          />
        </div>
      )}

      {/* Behavior score bar */}
      {metrics && (
        <div>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
            <span style={{ fontSize: 9, color: "var(--text-tertiary)" }}>Behavior Score</span>
            <span style={{
              fontSize: 9,
              fontFamily: "var(--font-mono)",
              color: metrics.behaviorScore > 60 ? "#10b981" : "#f59e0b",
            }}>
              {Math.round(metrics.behaviorScore)}%
            </span>
          </div>
          <div style={{ height: 4, borderRadius: 99, background: "var(--border-default)", overflow: "hidden" }}>
            <motion.div
              animate={{ width: `${metrics.behaviorScore}%` }}
              transition={{ duration: 0.5, ease: "easeOut" }}
              style={{
                height: "100%",
                background: metrics.behaviorScore > 60 ? "#10b981" : metrics.behaviorScore > 40 ? "#f59e0b" : "#ef4444",
                borderRadius: 99,
              }}
            />
          </div>
        </div>
      )}

      {!isEnabled && (
        <p style={{ fontSize: 11, color: "var(--text-tertiary)", textAlign: "center", marginTop: "auto" }}>
          Enable tracking to analyze behavior
        </p>
      )}
    </div>
  );
}

function BehaviorMetricTile({ label, value, status }: {
  label: string;
  value: string;
  status: "normal" | "high" | "low";
}) {
  const statusColor = status === "normal" ? "var(--text-primary)" : status === "high" ? "#f97316" : "#6366f1";
  return (
    <div style={{
      padding: "5px 8px",
      borderRadius: 6,
      background: "var(--bg-overlay)",
      border: "1px solid var(--border-subtle)",
    }}>
      <p style={{ fontSize: 9, color: "var(--text-tertiary)", marginBottom: 2 }}>{label}</p>
      <p style={{ fontSize: 11, fontWeight: 600, fontFamily: "var(--font-mono)", color: statusColor }}>{value}</p>
    </div>
  );
}
