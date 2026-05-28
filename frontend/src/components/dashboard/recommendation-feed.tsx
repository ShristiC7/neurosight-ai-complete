"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useDashboardStore } from "@/store/dashboard-store";
import type { Recommendation, RecommendationType } from "@/types";
import { apiClient } from "@/lib/api-client";

const TYPE_META: Record<RecommendationType, { icon: string; color: string }> = {
  take_break:    { icon: "☕", color: "#4f6ef7" },
  stretch:       { icon: "🧘", color: "#00d9c8" },
  hydrate:       { icon: "💧", color: "#3b82f6" },
  deep_work:     { icon: "🎯", color: "#a855f7" },
  light_task:    { icon: "📋", color: "#10b981" },
  sleep:         { icon: "🌙", color: "#6366f1" },
  exercise:      { icon: "⚡", color: "#f97316" },
  meditation:    { icon: "🧠", color: "#8b5cf6" },
  eye_rest:      { icon: "👁️", color: "#06b6d4" },
  posture_check: { icon: "🪑", color: "#84cc16" },
};

const PRIORITY_COLORS = {
  critical: "var(--danger)",
  high:     "var(--warning)",
  medium:   "var(--accent-primary)",
  low:      "var(--text-tertiary)",
};

export function RecommendationFeed() {
  const { recommendations, acceptRecommendation, dismissRecommendation } = useDashboardStore();

  const handleAccept = async (rec: Recommendation) => {
    acceptRecommendation(rec.id);
    try {
      await apiClient.patch(`/recommendations/${rec.id}/accept`);
    } catch {
      // Silent fail — local state already updated
    }
  };

  const handleDismiss = async (rec: Recommendation) => {
    dismissRecommendation(rec.id);
    try {
      await apiClient.patch(`/recommendations/${rec.id}/dismiss`);
    } catch {}
  };

  return (
    <div className="panel flex flex-col h-full" style={{ minHeight: 280 }}>
      <div className="flex items-center justify-between mb-4">
        <h3
          style={{
            fontFamily: "var(--font-display)",
            fontSize: 13,
            fontWeight: 600,
            letterSpacing: "0.05em",
            textTransform: "uppercase",
            color: "var(--text-primary)",
          }}
        >
          AI Coach
        </h3>
        {recommendations.length > 0 && (
          <span
            style={{
              background: "var(--accent-primary)",
              color: "white",
              borderRadius: 99,
              padding: "1px 8px",
              fontSize: 10,
              fontWeight: 600,
              fontFamily: "var(--font-mono)",
            }}
          >
            {recommendations.length}
          </span>
        )}
      </div>

      <div className="flex-1 overflow-y-auto space-y-3" style={{ maxHeight: 320 }}>
        <AnimatePresence mode="popLayout">
          {recommendations.length === 0 ? (
            <motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex flex-col items-center justify-center h-32"
            >
              <div style={{ fontSize: 32, marginBottom: 8 }}>🤖</div>
              <p style={{ color: "var(--text-tertiary)", fontSize: 12, textAlign: "center" }}>
                AI Coach is monitoring your patterns.<br />Recommendations will appear here.
              </p>
            </motion.div>
          ) : (
            recommendations.map((rec) => (
              <RecommendationCard
                key={rec.id}
                rec={rec}
                onAccept={() => handleAccept(rec)}
                onDismiss={() => handleDismiss(rec)}
              />
            ))
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

interface RecommendationCardProps {
  rec: Recommendation;
  onAccept: () => void;
  onDismiss: () => void;
}

function RecommendationCard({ rec, onAccept, onDismiss }: RecommendationCardProps) {
  const meta = TYPE_META[rec.type];

  return (
    <motion.div
      layout
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20, height: 0 }}
      transition={{ duration: 0.25 }}
      className="card"
      style={{
        borderLeft: `3px solid ${PRIORITY_COLORS[rec.priority]}`,
        borderLeftColor: PRIORITY_COLORS[rec.priority],
      }}
    >
      <div className="flex items-start gap-3">
        <span style={{ fontSize: 20, lineHeight: 1 }}>{meta.icon}</span>
        <div className="flex-1 min-w-0">
          <p
            style={{
              fontFamily: "var(--font-display)",
              fontSize: 12,
              fontWeight: 600,
              color: "var(--text-primary)",
              marginBottom: 2,
            }}
          >
            {rec.title}
          </p>
          <p
            style={{
              fontSize: 11,
              color: "var(--text-secondary)",
              lineHeight: 1.5,
            }}
          >
            {rec.message}
          </p>
          {rec.durationMinutes && (
            <p
              style={{
                fontSize: 10,
                color: "var(--text-tertiary)",
                marginTop: 4,
                fontFamily: "var(--font-mono)",
              }}
            >
              ⏱ {rec.durationMinutes} min
            </p>
          )}
        </div>
      </div>

      {rec.accepted === null && (
        <div className="flex gap-2 mt-3">
          <button
            onClick={onAccept}
            style={{
              flex: 1,
              padding: "5px 0",
              background: meta.color + "22",
              border: `1px solid ${meta.color}44`,
              borderRadius: 6,
              color: meta.color,
              fontSize: 11,
              fontWeight: 600,
              cursor: "pointer",
              transition: "all 150ms",
            }}
            onMouseEnter={(e) => {
              (e.currentTarget as HTMLButtonElement).style.background = meta.color + "33";
            }}
            onMouseLeave={(e) => {
              (e.currentTarget as HTMLButtonElement).style.background = meta.color + "22";
            }}
          >
            Accept
          </button>
          <button
            onClick={onDismiss}
            style={{
              padding: "5px 12px",
              background: "transparent",
              border: "1px solid var(--border-subtle)",
              borderRadius: 6,
              color: "var(--text-tertiary)",
              fontSize: 11,
              cursor: "pointer",
              transition: "all 150ms",
            }}
          >
            ✕
          </button>
        </div>
      )}

      {rec.accepted === true && (
        <p style={{ marginTop: 8, fontSize: 10, color: "var(--success)", fontFamily: "var(--font-mono)" }}>
          ✓ Accepted
        </p>
      )}
    </motion.div>
  );
}
