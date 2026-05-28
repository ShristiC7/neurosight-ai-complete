"use client";

import { motion } from "framer-motion";
import { format, parseISO } from "date-fns";
import type { ProductivityPrediction } from "@/types";

interface ProductivityPanelProps {
  prediction: ProductivityPrediction | null;
}

export function ProductivityPanel({ prediction }: ProductivityPanelProps) {
  const score = prediction?.productivityScore ?? 0;
  const burnout = prediction?.burnoutProbability ?? 0;
  const load = prediction?.cognitiveLoad ?? 0;

  const getScoreLabel = (s: number) => {
    if (s >= 80) return { label: "Peak", color: "#10b981" };
    if (s >= 60) return { label: "Good", color: "#4f6ef7" };
    if (s >= 40) return { label: "Moderate", color: "#f59e0b" };
    if (s >= 20) return { label: "Low", color: "#f97316" };
    return { label: "Critical", color: "#ef4444" };
  };

  const { label, color } = getScoreLabel(score);

  return (
    <div className="panel flex flex-col gap-3 h-full" style={{ minHeight: 200 }}>
      {/* Header */}
      <div className="flex items-center gap-2">
        <span style={{ fontSize: 16 }}>🎯</span>
        <span style={{
          fontFamily: "var(--font-display)",
          fontSize: 12,
          fontWeight: 600,
          letterSpacing: "0.05em",
          textTransform: "uppercase",
          color: "var(--text-primary)",
        }}>
          Productivity
        </span>
      </div>

      {/* Main score */}
      <div className="flex items-end justify-between">
        <div>
          <motion.p
            key={Math.round(score)}
            initial={{ y: -8, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            style={{
              fontSize: 36,
              fontWeight: 800,
              fontFamily: "var(--font-syne)",
              color,
              lineHeight: 1,
            }}
          >
            {Math.round(score)}
          </motion.p>
          <p style={{ fontSize: 10, color: "var(--text-tertiary)", marginTop: 2 }}>/ 100 score</p>
        </div>
        <span style={{
          fontSize: 11,
          fontWeight: 700,
          color,
          padding: "3px 10px",
          borderRadius: 99,
          background: `${color}15`,
          border: `1px solid ${color}30`,
        }}>
          {label}
        </span>
      </div>

      {/* Progress bars */}
      <div className="space-y-2">
        <ProgressBar label="Productivity" value={score} color={color} />
        <ProgressBar label="Burnout Risk" value={burnout * 100} color={burnout > 0.6 ? "#ef4444" : burnout > 0.3 ? "#f59e0b" : "#10b981"} />
        <ProgressBar label="Cognitive Load" value={load} color="#a855f7" />
      </div>

      {/* Predicted crash warning */}
      {prediction?.predictedCrashAt && (
        <div style={{
          padding: "6px 10px",
          borderRadius: 7,
          background: "rgba(239,68,68,0.08)",
          border: "1px solid rgba(239,68,68,0.2)",
        }}>
          <p style={{ fontSize: 10, color: "#ef4444", fontWeight: 600 }}>
            ⚠ Cognitive decline predicted
          </p>
          <p style={{ fontSize: 9, color: "var(--text-tertiary)", marginTop: 1 }}>
            ~{formatRelativeTime(prediction.predictedCrashAt)}
          </p>
        </div>
      )}

      {!prediction && (
        <p style={{ fontSize: 11, color: "var(--text-tertiary)", textAlign: "center", marginTop: "auto" }}>
          Start monitoring to see predictions
        </p>
      )}
    </div>
  );
}

function ProgressBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
        <span style={{ fontSize: 9, color: "var(--text-tertiary)" }}>{label}</span>
        <span style={{ fontSize: 9, fontFamily: "var(--font-mono)", color }}>{Math.round(value)}%</span>
      </div>
      <div style={{ height: 4, borderRadius: 99, background: "var(--border-default)", overflow: "hidden" }}>
        <motion.div
          animate={{ width: `${Math.min(value, 100)}%` }}
          transition={{ duration: 0.5, ease: "easeOut" }}
          style={{ height: "100%", background: color, borderRadius: 99 }}
        />
      </div>
    </div>
  );
}

function formatRelativeTime(isoStr: string): string {
  try {
    const diffMs = parseISO(isoStr).getTime() - Date.now();
    const diffMin = Math.round(Math.abs(diffMs) / 60000);
    if (diffMin < 60) return `${diffMin} min`;
    return `${Math.round(diffMin / 60)}h ${diffMin % 60}m`;
  } catch {
    return "--";
  }
}
