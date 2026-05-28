"use client";
// stress-panel, behavioral-panel, productivity-panel, burnout-gauge, focus-heatmap, topbar stubs

import type { VoiceStressMetrics, BehavioralMetrics, ProductivityPrediction } from "@/types";

// ============================================================
// StressPanel
// ============================================================
export function StressPanel({ metrics, isEnabled }: { metrics: VoiceStressMetrics | null; isEnabled: boolean }) {
  const emotion = metrics?.emotionState ?? "calm";
  const score = metrics?.stressScore ?? 0;

  const COLORS: Record<string, string> = {
    calm: "var(--success)", energetic: "var(--accent-secondary)",
    stressed: "var(--warning)", fatigued: "var(--danger)", anxious: "#a855f7",
  };

  return (
    <div className="panel flex flex-col h-full" style={{ minHeight: 160 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
        <p style={{ fontFamily: "var(--font-display)", fontSize: 11, letterSpacing: "0.08em", textTransform: "uppercase", color: "var(--text-tertiary)" }}>Voice Stress</p>
        <span style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: isEnabled ? "var(--success)" : "var(--text-tertiary)" }}>
          {isEnabled ? "● LIVE" : "○ OFF"}
        </span>
      </div>
      <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 8 }}>
        <span style={{ fontFamily: "var(--font-display)", fontSize: "2.2rem", fontWeight: 700, color: COLORS[emotion] ?? "var(--text-primary)", lineHeight: 1 }}>
          {Math.round(score)}<span style={{ fontSize: 12, color: "var(--text-secondary)" }}> / 100</span>
        </span>
        <span style={{ fontSize: 11, color: COLORS[emotion] ?? "var(--text-secondary)", textTransform: "capitalize", fontWeight: 600 }}>
          {emotion}
        </span>
        {metrics && (
          <div style={{ fontSize: 10, color: "var(--text-tertiary)" }}>
            Energy: {metrics.speechEnergy.toFixed(3)} · Pitch Δ: {Math.round(metrics.pitchVariance)}
          </div>
        )}
        {!isEnabled && <p style={{ fontSize: 11, color: "var(--text-tertiary)" }}>Enable microphone to analyze stress</p>}
      </div>
    </div>
  );
}

// ============================================================
// BehavioralPanel
// ============================================================
export function BehavioralPanel({ metrics, isEnabled }: { metrics: BehavioralMetrics | null; isEnabled: boolean }) {
  return (
    <div className="panel flex flex-col h-full" style={{ minHeight: 160 }}>
      <p style={{ fontFamily: "var(--font-display)", fontSize: 11, letterSpacing: "0.08em", textTransform: "uppercase", color: "var(--text-tertiary)", marginBottom: 12 }}>Behavioral</p>
      <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 8 }}>
        <span style={{ fontFamily: "var(--font-display)", fontSize: "2.2rem", fontWeight: 700, color: "var(--accent-primary)", lineHeight: 1 }}>
          {metrics ? Math.round(metrics.behaviorScore) : "--"}<span style={{ fontSize: 12, color: "var(--text-secondary)" }}> / 100</span>
        </span>
        {metrics && (
          <>
            <Row label="Typing" value={`${Math.round(metrics.typingSpeed)} WPM`} />
            <Row label="Error Rate" value={`${metrics.errorRate.toFixed(1)}/min`} />
            <Row label="Anomaly" value={`${Math.round(metrics.anomalyScore * 100)}%`} />
            <Row label="Focus" value={`${Math.round(metrics.focusSessionDuration)}m`} />
          </>
        )}
        {!isEnabled && <p style={{ fontSize: 11, color: "var(--text-tertiary)" }}>Enable keyboard tracking for behavioral analytics</p>}
      </div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between" }}>
      <span style={{ fontSize: 10, color: "var(--text-tertiary)" }}>{label}</span>
      <span style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--text-secondary)" }}>{value}</span>
    </div>
  );
}

// ============================================================
// ProductivityPanel
// ============================================================
export function ProductivityPanel({ prediction }: { prediction: ProductivityPrediction | null }) {
  const score = prediction?.productivityScore ?? 0;
  const burnout = prediction ? Math.round(prediction.burnoutProbability * 100) : 0;

  return (
    <div className="panel flex flex-col h-full" style={{ minHeight: 160 }}>
      <p style={{ fontFamily: "var(--font-display)", fontSize: 11, letterSpacing: "0.08em", textTransform: "uppercase", color: "var(--text-tertiary)", marginBottom: 12 }}>Productivity</p>
      <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 8 }}>
        <span style={{ fontFamily: "var(--font-display)", fontSize: "2.2rem", fontWeight: 700, color: "var(--accent-secondary)", lineHeight: 1 }}>
          {Math.round(score)}<span style={{ fontSize: 12, color: "var(--text-secondary)" }}> / 100</span>
        </span>
        <Row label="Burnout Risk" value={`${burnout}%`} />
        {prediction?.cognitiveLoad !== undefined && (
          <Row label="Cognitive Load" value={`${Math.round(prediction.cognitiveLoad)}%`} />
        )}
        {prediction?.confidence !== undefined && (
          <Row label="AI Confidence" value={`${Math.round(prediction.confidence * 100)}%`} />
        )}
      </div>
    </div>
  );
}
