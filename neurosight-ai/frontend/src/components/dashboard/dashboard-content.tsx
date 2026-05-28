"use client";

import { useDashboardStore } from "@/store/dashboard-store";
import { useWSEvent } from "@/hooks/use-websocket";
import { useFatigueDetection } from "@/hooks/use-fatigue-detection";
import { useVoiceStress } from "@/hooks/use-voice-stress";
import { useBehavioralAnalytics } from "@/hooks/use-behavioral-analytics";
import { useAuthStore } from "@/store/auth-store";
import { FatiguePanel } from "@/components/fatigue/fatigue-panel";
import { StressPanel } from "@/components/audio/stress-panel";
import { ProductivityPanel } from "@/components/dashboard/productivity-panel";
import { BehavioralPanel } from "@/components/behavioral/behavioral-panel";
import { RecommendationFeed } from "@/components/dashboard/recommendation-feed";
import { CognitiveScoreRing } from "@/components/dashboard/cognitive-score-ring";
import { SessionControls } from "@/components/dashboard/session-controls";
import { TimelineChart } from "@/components/charts/timeline-chart";
import { BurnoutRiskGauge } from "@/components/charts/burnout-risk-gauge";
import { FocusHeatmap } from "@/components/charts/focus-heatmap";
import type { FatigueMetrics, VoiceStressMetrics, BehavioralMetrics, ProductivityPrediction, Recommendation } from "@/types";
import { motion } from "framer-motion";

const containerVariants = {
  hidden: {},
  show: {
    transition: { staggerChildren: 0.06, delayChildren: 0.1 },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.35, ease: [0.4, 0, 0.2, 1] } },
};

export function DashboardContent() {
  const { user } = useAuthStore();
  const {
    dashboard,
    latestFatigue,
    latestStress,
    latestBehavioral,
    latestPrediction,
    fatigueHistory,
    stressHistory,
    productivityHistory,
    isMonitoringActive,
    isCameraEnabled,
    isMicEnabled,
    isKeyboardTracking,
    updateFatigue,
    updateStress,
    updateBehavioral,
    updatePrediction,
    addRecommendation,
  } = useDashboardStore();

  const sessionId = dashboard.activeSession?.id ?? "no-session";
  const userId = user?.id ?? "unknown";

  // --- Real-time sensor hooks ---
  useFatigueDetection({
    sessionId,
    userId,
    enabled: isCameraEnabled && isMonitoringActive,
    onFatigueUpdate: updateFatigue,
    onCriticalAlert: () => console.warn("CRITICAL FATIGUE ALERT"),
  });

  useVoiceStress({
    sessionId,
    userId,
    enabled: isMicEnabled && isMonitoringActive,
    onStressUpdate: updateStress,
  });

  useBehavioralAnalytics({
    sessionId,
    userId,
    enabled: isKeyboardTracking && isMonitoringActive,
    onBehavioralUpdate: updateBehavioral,
  });

  // --- WebSocket event listeners ---
  useWSEvent<FatigueMetrics>("fatigue:update", updateFatigue);
  useWSEvent<VoiceStressMetrics>("stress:update", updateStress);
  useWSEvent<BehavioralMetrics>("behavioral:update", updateBehavioral);
  useWSEvent<ProductivityPrediction>("prediction:update", updatePrediction);
  useWSEvent<Recommendation>("recommendation:new", addRecommendation);

  const overallScore = Math.round(
    (100 - dashboard.fatigueScore) * 0.35 +
    (100 - dashboard.stressScore) * 0.25 +
    dashboard.productivityScore * 0.25 +
    dashboard.focusLevel * 0.15
  );

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="show"
      className="space-y-6"
    >
      {/* ---- Row 1: Score rings + Session Controls ---- */}
      <motion.div
        variants={itemVariants}
        className="grid grid-cols-1 lg:grid-cols-4 gap-4"
      >
        <div className="lg:col-span-1">
          <CognitiveScoreRing
            score={overallScore}
            fatigueScore={dashboard.fatigueScore}
            stressScore={dashboard.stressScore}
            productivityScore={dashboard.productivityScore}
            isLive={isMonitoringActive}
          />
        </div>

        <div className="lg:col-span-2 grid grid-cols-3 gap-4">
          <MetricCard
            label="Fatigue"
            value={dashboard.fatigueScore}
            unit="%"
            color={getFatigueColor(dashboard.fatigueScore)}
            trend={fatigueHistory.slice(-2)}
            inverse
          />
          <MetricCard
            label="Stress"
            value={dashboard.stressScore}
            unit="%"
            color={getStressColor(dashboard.stressScore)}
            trend={stressHistory.slice(-2)}
            inverse
          />
          <MetricCard
            label="Productivity"
            value={dashboard.productivityScore}
            unit="%"
            color="var(--accent-secondary)"
            trend={productivityHistory.slice(-2)}
          />
        </div>

        <div className="lg:col-span-1">
          <SessionControls />
        </div>
      </motion.div>

      {/* ---- Row 2: Timeline + Burnout Gauge ---- */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <motion.div variants={itemVariants} className="lg:col-span-2">
          <TimelineChart
            fatigueData={fatigueHistory}
            stressData={stressHistory}
            productivityData={productivityHistory}
          />
        </motion.div>
        <motion.div variants={itemVariants}>
          <BurnoutRiskGauge
            burnoutRisk={dashboard.burnoutRisk}
            focusLevel={dashboard.focusLevel}
            currentStreak={dashboard.currentStreak}
            prediction={latestPrediction}
          />
        </motion.div>
      </div>

      {/* ---- Row 3: Sensor panels ---- */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <motion.div variants={itemVariants}>
          <FatiguePanel metrics={latestFatigue} isEnabled={isCameraEnabled} />
        </motion.div>
        <motion.div variants={itemVariants}>
          <StressPanel metrics={latestStress} isEnabled={isMicEnabled} />
        </motion.div>
        <motion.div variants={itemVariants}>
          <ProductivityPanel prediction={latestPrediction} />
        </motion.div>
        <motion.div variants={itemVariants}>
          <BehavioralPanel metrics={latestBehavioral} isEnabled={isKeyboardTracking} />
        </motion.div>
      </div>

      {/* ---- Row 4: Heatmap + Recommendations ---- */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
        <motion.div variants={itemVariants} className="lg:col-span-3">
          <FocusHeatmap />
        </motion.div>
        <motion.div variants={itemVariants} className="lg:col-span-2">
          <RecommendationFeed />
        </motion.div>
      </div>
    </motion.div>
  );
}

// -----------------------------------------------------------
// Metric Card (small KPI tiles)
// -----------------------------------------------------------
interface MetricCardProps {
  label: string;
  value: number;
  unit: string;
  color: string;
  trend: { value: number }[];
  inverse?: boolean;
}

function MetricCard({ label, value, unit, color, trend, inverse }: MetricCardProps) {
  const delta = trend.length >= 2 ? trend[trend.length - 1].value - trend[trend.length - 2].value : 0;
  const isGood = inverse ? delta <= 0 : delta >= 0;

  return (
    <div className="panel flex flex-col justify-between h-full min-h-[100px]">
      <p style={{ color: "var(--text-secondary)", fontFamily: "var(--font-display)", fontSize: "11px", letterSpacing: "0.08em", textTransform: "uppercase" }}>
        {label}
      </p>
      <div className="flex items-end justify-between mt-2">
        <span
          style={{
            fontFamily: "var(--font-display)",
            fontSize: "2rem",
            fontWeight: 700,
            color,
            lineHeight: 1,
          }}
        >
          {Math.round(value)}
          <span style={{ fontSize: "0.875rem", color: "var(--text-secondary)", marginLeft: "2px" }}>{unit}</span>
        </span>
        {Math.abs(delta) > 0.5 && (
          <span style={{ fontSize: "11px", color: isGood ? "var(--success)" : "var(--danger)" }}>
            {delta > 0 ? "↑" : "↓"} {Math.abs(Math.round(delta))}
          </span>
        )}
      </div>
    </div>
  );
}

function getFatigueColor(score: number): string {
  if (score < 30) return "var(--fatigue-alert)";
  if (score < 50) return "var(--fatigue-mild)";
  if (score < 70) return "var(--fatigue-moderate)";
  if (score < 85) return "var(--fatigue-severe)";
  return "var(--fatigue-critical)";
}

function getStressColor(score: number): string {
  if (score < 30) return "var(--success)";
  if (score < 60) return "var(--warning)";
  return "var(--danger)";
}
