"use client";

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { format, parseISO } from "date-fns";
import type { TimeSeriesDataPoint } from "@/types";
import { useMemo } from "react";

interface TimelineChartProps {
  fatigueData: TimeSeriesDataPoint[];
  stressData: TimeSeriesDataPoint[];
  productivityData: TimeSeriesDataPoint[];
}

interface ChartPoint {
  time: string;
  fatigue: number;
  stress: number;
  productivity: number;
}

export function TimelineChart({
  fatigueData,
  stressData,
  productivityData,
}: TimelineChartProps) {
  const chartData = useMemo<ChartPoint[]>(() => {
    const length = Math.max(fatigueData.length, stressData.length, productivityData.length);
    return Array.from({ length }, (_, i) => ({
      time: fatigueData[i]?.timestamp ?? stressData[i]?.timestamp ?? new Date().toISOString(),
      fatigue: fatigueData[i]?.value ?? 0,
      stress: stressData[i]?.value ?? 0,
      productivity: productivityData[i]?.value ?? 0,
    }));
  }, [fatigueData, stressData, productivityData]);

  if (chartData.length === 0) {
    return (
      <div className="panel flex items-center justify-center" style={{ minHeight: 240 }}>
        <div className="text-center">
          <div
            style={{
              width: 40,
              height: 40,
              borderRadius: 10,
              background: "rgba(79,110,247,0.1)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              margin: "0 auto 12px",
            }}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#4f6ef7" strokeWidth={1.5}>
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
            </svg>
          </div>
          <p style={{ color: "var(--text-secondary)", fontSize: 13 }}>
            Start monitoring to see real-time trends
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="panel" style={{ minHeight: 240 }}>
      <div className="flex items-center justify-between mb-4">
        <h3
          style={{
            fontFamily: "var(--font-display)",
            fontSize: 13,
            fontWeight: 600,
            color: "var(--text-primary)",
            letterSpacing: "0.05em",
            textTransform: "uppercase",
          }}
        >
          Cognitive Timeline
        </h3>
        <span
          style={{
            fontSize: 11,
            color: "var(--text-tertiary)",
            fontFamily: "var(--font-mono)",
          }}
        >
          Last {chartData.length} readings
        </span>
      </div>

      <ResponsiveContainer width="100%" height={180}>
        <AreaChart data={chartData} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
          <defs>
            <linearGradient id="gradFatigue" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#ef4444" stopOpacity={0.2} />
              <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="gradStress" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.15} />
              <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="gradProductivity" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#00d9c8" stopOpacity={0.2} />
              <stop offset="95%" stopColor="#00d9c8" stopOpacity={0} />
            </linearGradient>
          </defs>

          <CartesianGrid
            strokeDasharray="3 3"
            stroke="rgba(99,120,220,0.06)"
            vertical={false}
          />

          <XAxis
            dataKey="time"
            tickFormatter={(val) => {
              try {
                return format(parseISO(val), "HH:mm");
              } catch {
                return "";
              }
            }}
            tick={{ fill: "var(--text-tertiary)", fontSize: 10, fontFamily: "var(--font-mono)" }}
            axisLine={false}
            tickLine={false}
            interval="preserveStartEnd"
          />

          <YAxis
            domain={[0, 100]}
            tick={{ fill: "var(--text-tertiary)", fontSize: 10, fontFamily: "var(--font-mono)" }}
            axisLine={false}
            tickLine={false}
            tickCount={5}
          />

          <Tooltip
            content={({ active, payload, label }) => {
              if (!active || !payload?.length) return null;
              let timeStr = label;
              try {
                timeStr = format(parseISO(label as string), "HH:mm:ss");
              } catch {}
              return (
                <div
                  style={{
                    background: "var(--bg-overlay)",
                    border: "1px solid var(--border-default)",
                    borderRadius: 8,
                    padding: "8px 12px",
                    fontSize: 12,
                    fontFamily: "var(--font-mono)",
                  }}
                >
                  <p style={{ color: "var(--text-secondary)", marginBottom: 6 }}>{timeStr}</p>
                  {payload.map((entry) => (
                    <div key={entry.dataKey} style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 2 }}>
                      <span style={{ width: 8, height: 8, borderRadius: "50%", background: entry.color as string }} />
                      <span style={{ color: "var(--text-secondary)", textTransform: "capitalize" }}>{entry.dataKey}:</span>
                      <span style={{ color: "white", fontWeight: 600 }}>{Math.round(entry.value as number)}</span>
                    </div>
                  ))}
                </div>
              );
            }}
          />

          <Area
            type="monotone"
            dataKey="fatigue"
            stroke="#ef4444"
            strokeWidth={1.5}
            fill="url(#gradFatigue)"
            dot={false}
            activeDot={{ r: 3, fill: "#ef4444", strokeWidth: 0 }}
          />
          <Area
            type="monotone"
            dataKey="stress"
            stroke="#f59e0b"
            strokeWidth={1.5}
            fill="url(#gradStress)"
            dot={false}
            activeDot={{ r: 3, fill: "#f59e0b", strokeWidth: 0 }}
          />
          <Area
            type="monotone"
            dataKey="productivity"
            stroke="#00d9c8"
            strokeWidth={1.5}
            fill="url(#gradProductivity)"
            dot={false}
            activeDot={{ r: 3, fill: "#00d9c8", strokeWidth: 0 }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
