"use client";

import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";
import type { HeatmapCell } from "@/types";
import { motion } from "framer-motion";

const HOURS = Array.from({ length: 24 }, (_, i) => i);
const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

// Fallback synthetic data for empty state
const SYNTHETIC_DATA: HeatmapCell[] = DAYS.flatMap((_, dayIdx) =>
  HOURS.map((hour) => {
    // Simulate realistic work patterns
    const isWorkHour = hour >= 9 && hour <= 18;
    const isWeekend = dayIdx >= 5;
    const isPeakMorning = hour >= 9 && hour <= 11;
    const isPostLunch = hour >= 13 && hour <= 15;

    let value = 0;
    if (!isWeekend && isWorkHour) {
      value = isPeakMorning ? 70 + Math.random() * 25
        : isPostLunch ? 40 + Math.random() * 20
        : 50 + Math.random() * 25;
    } else if (!isWeekend && (hour === 8 || hour === 19)) {
      value = 20 + Math.random() * 20;
    }

    return {
      day: dayIdx,
      hour,
      value: Math.round(value),
      label: `${DAYS[dayIdx]} ${hour}:00`,
    };
  })
);

function getCellColor(value: number): string {
  if (value === 0) return "rgba(255,255,255,0.03)";
  if (value < 20) return "rgba(79,110,247,0.08)";
  if (value < 40) return "rgba(79,110,247,0.18)";
  if (value < 60) return "rgba(79,110,247,0.35)";
  if (value < 80) return "rgba(79,110,247,0.6)";
  return "rgba(79,110,247,0.85)";
}

export function FocusHeatmap() {
  const { data: serverData, isLoading } = useQuery<HeatmapCell[]>({
    queryKey: ["focus-heatmap"],
    queryFn: () => apiClient.get<HeatmapCell[]>("/analytics/focus-heatmap"),
    staleTime: 5 * 60 * 1000,
    retry: false,
  });

  const cells = serverData ?? SYNTHETIC_DATA;
  const cellMap = new Map(cells.map((c) => [`${c.day}-${c.hour}`, c]));

  // Label hours every 3
  const hourLabels = HOURS.filter((h) => h % 3 === 0);

  return (
    <div className="panel" style={{ minHeight: 220 }}>
      <div className="flex items-center justify-between mb-4">
        <h3 style={{
          fontFamily: "var(--font-display)",
          fontSize: 13,
          fontWeight: 600,
          letterSpacing: "0.05em",
          textTransform: "uppercase",
          color: "var(--text-primary)",
        }}>
          Weekly Focus Heatmap
        </h3>
        {!serverData && (
          <span style={{ fontSize: 10, color: "var(--text-tertiary)", fontFamily: "var(--font-mono)" }}>
            Simulated pattern
          </span>
        )}
      </div>

      <div style={{ overflowX: "auto" }}>
        <div style={{ minWidth: 520 }}>
          {/* Hour labels */}
          <div style={{ display: "flex", paddingLeft: 32, marginBottom: 4 }}>
            {HOURS.map((h) => (
              <div
                key={h}
                style={{
                  flex: 1,
                  textAlign: "center",
                  fontSize: 8,
                  color: hourLabels.includes(h) ? "var(--text-tertiary)" : "transparent",
                  fontFamily: "var(--font-mono)",
                }}
              >
                {h === 0 ? "12a" : h < 12 ? `${h}a` : h === 12 ? "12p" : `${h - 12}p`}
              </div>
            ))}
          </div>

          {/* Grid */}
          {DAYS.map((day, dayIdx) => (
            <div
              key={day}
              style={{ display: "flex", alignItems: "center", marginBottom: 3, gap: 2 }}
            >
              <div style={{
                width: 28,
                fontSize: 10,
                color: "var(--text-tertiary)",
                fontFamily: "var(--font-mono)",
                textAlign: "right",
                paddingRight: 4,
                flexShrink: 0,
              }}>
                {day}
              </div>
              {HOURS.map((hour) => {
                const cell = cellMap.get(`${dayIdx}-${hour}`);
                const value = cell?.value ?? 0;
                return (
                  <motion.div
                    key={hour}
                    whileHover={{ scale: 1.3, zIndex: 10 }}
                    title={`${day} ${hour}:00 — Focus: ${value}%`}
                    style={{
                      flex: 1,
                      height: 14,
                      borderRadius: 2,
                      background: getCellColor(value),
                      border: "1px solid rgba(255,255,255,0.03)",
                      cursor: "pointer",
                      transition: "background 300ms",
                    }}
                  />
                );
              })}
            </div>
          ))}

          {/* Legend */}
          <div style={{ display: "flex", alignItems: "center", gap: 6, paddingLeft: 32, marginTop: 10 }}>
            <span style={{ fontSize: 9, color: "var(--text-tertiary)" }}>Less</span>
            {[0, 20, 40, 60, 80, 100].map((v) => (
              <div
                key={v}
                style={{
                  width: 12,
                  height: 12,
                  borderRadius: 2,
                  background: getCellColor(v),
                  border: "1px solid rgba(255,255,255,0.03)",
                }}
              />
            ))}
            <span style={{ fontSize: 9, color: "var(--text-tertiary)" }}>More</span>
          </div>
        </div>
      </div>
    </div>
  );
}
