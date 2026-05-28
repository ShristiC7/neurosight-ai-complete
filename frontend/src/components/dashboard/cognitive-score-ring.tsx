"use client";

import { motion } from "framer-motion";
import { useMemo } from "react";

interface CognitiveScoreRingProps {
  score: number;
  fatigueScore: number;
  stressScore: number;
  productivityScore: number;
  isLive: boolean;
}

export function CognitiveScoreRing({
  score,
  fatigueScore,
  stressScore,
  productivityScore,
  isLive,
}: CognitiveScoreRingProps) {
  const { color, label } = useMemo(() => {
    if (score >= 80) return { color: "#10b981", label: "Optimal" };
    if (score >= 60) return { color: "#4f6ef7", label: "Focused" };
    if (score >= 40) return { color: "#f59e0b", label: "Declining" };
    if (score >= 20) return { color: "#f97316", label: "Fatigued" };
    return { color: "#ef4444", label: "Critical" };
  }, [score]);

  // SVG ring parameters
  const R = 56;
  const STROKE = 8;
  const cx = 70;
  const cy = 70;
  const circumference = 2 * Math.PI * R;
  const dashOffset = circumference * (1 - score / 100);

  // Inner rings for individual metrics
  const innerRings = [
    { r: 44, score: 100 - fatigueScore, color: "#ef4444", label: "Fatigue" },
    { r: 33, score: 100 - stressScore, color: "#f59e0b", label: "Stress" },
    { r: 22, score: productivityScore, color: "#00d9c8", label: "Focus" },
  ];

  return (
    <div
      className="panel flex flex-col items-center justify-center h-full"
      style={{ minHeight: 160, position: "relative" }}
    >
      {/* Live badge */}
      {isLive && (
        <div
          className="absolute top-3 right-3 flex items-center gap-1.5"
          style={{ fontSize: 10, color: "var(--success)" }}
        >
          <span className="status-dot active" />
          <span style={{ fontFamily: "var(--font-mono)", letterSpacing: "0.05em" }}>LIVE</span>
        </div>
      )}

      <div className="relative">
        <svg width={140} height={140} viewBox="0 0 140 140">
          {/* Background rings */}
          <circle cx={cx} cy={cy} r={R} fill="none" stroke="rgba(79,110,247,0.08)" strokeWidth={STROKE} />

          {/* Main score arc */}
          <motion.circle
            cx={cx}
            cy={cy}
            r={R}
            fill="none"
            stroke={color}
            strokeWidth={STROKE}
            strokeLinecap="round"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: dashOffset }}
            transition={{ duration: 1, ease: "easeOut" }}
            style={{
              transformOrigin: `${cx}px ${cy}px`,
              transform: "rotate(-90deg)",
              filter: `drop-shadow(0 0 8px ${color}60)`,
            }}
          />

          {/* Inner rings */}
          {innerRings.map(({ r, score: s, color: c }) => {
            const circ = 2 * Math.PI * r;
            const offset = circ * (1 - s / 100);
            return (
              <g key={r}>
                <circle
                  cx={cx} cy={cy} r={r}
                  fill="none"
                  stroke="rgba(255,255,255,0.04)"
                  strokeWidth={4}
                />
                <motion.circle
                  cx={cx} cy={cy} r={r}
                  fill="none"
                  stroke={c}
                  strokeWidth={4}
                  strokeLinecap="round"
                  strokeDasharray={circ}
                  initial={{ strokeDashoffset: circ }}
                  animate={{ strokeDashoffset: offset }}
                  transition={{ duration: 0.8, ease: "easeOut", delay: 0.2 }}
                  style={{
                    transformOrigin: `${cx}px ${cy}px`,
                    transform: "rotate(-90deg)",
                    opacity: 0.7,
                  }}
                />
              </g>
            );
          })}

          {/* Center score */}
          <text
            x={cx}
            y={cy - 6}
            textAnchor="middle"
            dominantBaseline="middle"
            fill="white"
            fontSize={28}
            fontWeight={700}
            fontFamily="var(--font-syne)"
          >
            {Math.round(score)}
          </text>
          <text
            x={cx}
            y={cy + 14}
            textAnchor="middle"
            fill={color}
            fontSize={9}
            fontFamily="var(--font-dm-sans)"
            letterSpacing={1.5}
          >
            {label.toUpperCase()}
          </text>
        </svg>
      </div>

      <p
        style={{
          marginTop: 4,
          fontSize: 11,
          color: "var(--text-tertiary)",
          fontFamily: "var(--font-display)",
          letterSpacing: "0.06em",
          textTransform: "uppercase",
        }}
      >
        Cognitive Score
      </p>
    </div>
  );
}
