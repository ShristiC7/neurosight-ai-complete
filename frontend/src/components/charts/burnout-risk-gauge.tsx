"use client";

import { motion } from "framer-motion";
import { format, parseISO } from "date-fns";
import type { ProductivityPrediction } from "@/types";

interface BurnoutRiskGaugeProps {
  burnoutRisk: number;
  focusLevel: number;
  currentStreak: number;
  prediction: ProductivityPrediction | null;
}

export function BurnoutRiskGauge({
  burnoutRisk,
  focusLevel,
  currentStreak,
  prediction,
}: BurnoutRiskGaugeProps) {
  const getRiskLevel = (risk: number) => {
    if (risk < 20) return { label: "Minimal", color: "#10b981" };
    if (risk < 40) return { label: "Low", color: "#4f6ef7" };
    if (risk < 60) return { label: "Moderate", color: "#f59e0b" };
    if (risk < 80) return { label: "High", color: "#f97316" };
    return { label: "Critical", color: "#ef4444" };
  };

  const { label, color } = getRiskLevel(burnoutRisk);

  // Gauge arc parameters
  const R = 52;
  const cx = 70;
  const cy = 80;
  // Semi-circle arc (180 degrees)
  const startAngle = -180;
  const endAngle = 0;
  const riskAngle = startAngle + (burnoutRisk / 100) * 180;

  function polarToCartesian(cx: number, cy: number, r: number, angleDeg: number) {
    const angleRad = (angleDeg * Math.PI) / 180;
    return {
      x: cx + r * Math.cos(angleRad),
      y: cy + r * Math.sin(angleRad),
    };
  }

  function describeArc(cx: number, cy: number, r: number, startAngle: number, endAngle: number) {
    const start = polarToCartesian(cx, cy, r, endAngle);
    const end = polarToCartesian(cx, cy, r, startAngle);
    const largeArc = endAngle - startAngle <= 180 ? "0" : "1";
    return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 0 ${end.x} ${end.y}`;
  }

  const bgPath = describeArc(cx, cy, R, -180, 0);
  const riskPath = describeArc(cx, cy, R, -180, riskAngle);
  const needle = polarToCartesian(cx, cy, R - 8, riskAngle);

  return (
    <div className="panel flex flex-col gap-3" style={{ minHeight: 240 }}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 style={{
          fontFamily: "var(--font-display)",
          fontSize: 13,
          fontWeight: 600,
          letterSpacing: "0.05em",
          textTransform: "uppercase",
          color: "var(--text-primary)",
        }}>
          Burnout Risk
        </h3>
        <span style={{
          fontSize: 11,
          padding: "2px 8px",
          borderRadius: 99,
          background: `${color}20`,
          border: `1px solid ${color}40`,
          color,
          fontWeight: 600,
        }}>
          {label}
        </span>
      </div>

      {/* Gauge SVG */}
      <div className="flex justify-center">
        <svg width={140} height={90} viewBox="0 0 140 90">
          {/* Background track */}
          <path d={bgPath} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={10} strokeLinecap="round" />

          {/* Color zones */}
          {[
            { start: -180, end: -144, color: "#10b981" },
            { start: -144, end: -108, color: "#4f6ef7" },
            { start: -108, end: -72, color: "#f59e0b" },
            { start: -72,  end: -36,  color: "#f97316" },
            { start: -36,  end: 0,    color: "#ef4444" },
          ].map((zone, i) => (
            <path
              key={i}
              d={describeArc(cx, cy, R, zone.start, zone.end)}
              fill="none"
              stroke={zone.color}
              strokeWidth={10}
              strokeLinecap="butt"
              opacity={0.25}
            />
          ))}

          {/* Active risk arc */}
          {burnoutRisk > 0 && (
            <motion.path
              d={riskPath}
              fill="none"
              stroke={color}
              strokeWidth={10}
              strokeLinecap="round"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 0.8, ease: "easeOut" }}
              style={{ filter: `drop-shadow(0 0 6px ${color}80)` }}
            />
          )}

          {/* Needle */}
          <motion.line
            x1={cx}
            y1={cy}
            animate={{ x2: needle.x, y2: needle.y }}
            transition={{ duration: 0.6, ease: "easeOut" }}
            stroke={color}
            strokeWidth={2}
            strokeLinecap="round"
          />
          <circle cx={cx} cy={cy} r={5} fill={color} />

          {/* Score */}
          <text x={cx} y={cy - 16} textAnchor="middle" fill="white" fontSize={22} fontWeight={700} fontFamily="var(--font-syne)">
            {Math.round(burnoutRisk)}%
          </text>
        </svg>
      </div>

      {/* Prediction details */}
      {prediction && (
        <div className="space-y-2">
          {prediction.recommendedBreakAt && (
            <InfoRow
              label="Break Recommended"
              value={formatRelativeTime(prediction.recommendedBreakAt)}
              color="#4f6ef7"
            />
          )}
          {prediction.focusWindowStart && (
            <InfoRow
              label="Peak Focus Window"
              value={`${formatTime(prediction.focusWindowStart)} – ${formatTime(prediction.focusWindowEnd ?? "")}`}
              color="#00d9c8"
            />
          )}
          <InfoRow
            label="Confidence"
            value={`${Math.round(prediction.confidence * 100)}%`}
            color="var(--text-secondary)"
          />
        </div>
      )}

      {/* Focus streak */}
      <div
        style={{
          padding: "8px 10px",
          borderRadius: 8,
          background: "var(--bg-overlay)",
          border: "1px solid var(--border-subtle)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <span style={{ fontSize: 10, color: "var(--text-tertiary)" }}>Focus Level</span>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ width: 60, height: 4, borderRadius: 99, background: "var(--border-default)", overflow: "hidden" }}>
            <motion.div
              animate={{ width: `${focusLevel}%` }}
              style={{ height: "100%", background: "#00d9c8", borderRadius: 99 }}
              transition={{ duration: 0.5 }}
            />
          </div>
          <span style={{ fontSize: 11, fontWeight: 600, fontFamily: "var(--font-mono)", color: "#00d9c8" }}>
            {Math.round(focusLevel)}%
          </span>
        </div>
      </div>
    </div>
  );
}

function InfoRow({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
      <span style={{ fontSize: 10, color: "var(--text-tertiary)" }}>{label}</span>
      <span style={{ fontSize: 11, fontWeight: 600, color }}>{value}</span>
    </div>
  );
}

function formatTime(isoStr: string): string {
  try {
    return format(parseISO(isoStr), "HH:mm");
  } catch {
    return "--:--";
  }
}

function formatRelativeTime(isoStr: string): string {
  try {
    const diffMs = parseISO(isoStr).getTime() - Date.now();
    const diffMin = Math.round(diffMs / 60000);
    if (diffMin <= 0) return "Now";
    if (diffMin < 60) return `In ${diffMin}m`;
    return `In ${Math.round(diffMin / 60)}h`;
  } catch {
    return "--";
  }
}
