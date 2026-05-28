import { describe, it, expect, beforeEach } from "vitest";
import { useDashboardStore } from "../dashboard-store";
import type { FatigueMetrics, VoiceStressMetrics, BehavioralMetrics, Recommendation } from "@/types";

const makeFatigue = (score: number): FatigueMetrics => ({
  id: "id-1", userId: "u1", sessionId: "s1",
  timestamp: new Date().toISOString(),
  blinkRate: 15, eyeAspectRatio: 0.3, mouthAspectRatio: 0.2,
  headTiltAngle: 0, gazeDrift: 0, fatigueScore: score,
  drowsinessLevel: score < 30 ? "alert" : score < 60 ? "moderate" : "severe",
  confidence: 0.9,
});

const makeRec = (id: string): Recommendation => ({
  id, userId: "u1", sessionId: "s1",
  timestamp: new Date().toISOString(),
  type: "take_break", priority: "medium",
  title: "Take a Break", message: "Rest up.",
  accepted: null,
  expiresAt: new Date(Date.now() + 900_000).toISOString(),
  metadata: {},
});

describe("Dashboard Store", () => {
  beforeEach(() => {
    useDashboardStore.getState().reset();
  });

  // ── Initial state ────────────────────────────────────────────────────────

  it("initializes with zero fatigue score", () => {
    expect(useDashboardStore.getState().dashboard.fatigueScore).toBe(0);
  });

  it("initializes with monitoring inactive", () => {
    expect(useDashboardStore.getState().isMonitoringActive).toBe(false);
  });

  it("initializes with empty recommendations", () => {
    expect(useDashboardStore.getState().recommendations).toHaveLength(0);
  });

  it("initializes with empty fatigue history", () => {
    expect(useDashboardStore.getState().fatigueHistory).toHaveLength(0);
  });

  // ── Fatigue updates ──────────────────────────────────────────────────────

  it("updates fatigue score", () => {
    useDashboardStore.getState().updateFatigue(makeFatigue(45));
    expect(useDashboardStore.getState().dashboard.fatigueScore).toBe(45);
  });

  it("appends to fatigue history", () => {
    useDashboardStore.getState().updateFatigue(makeFatigue(30));
    useDashboardStore.getState().updateFatigue(makeFatigue(50));
    expect(useDashboardStore.getState().fatigueHistory).toHaveLength(2);
  });

  it("stores latest fatigue metrics", () => {
    const m = makeFatigue(70);
    useDashboardStore.getState().updateFatigue(m);
    expect(useDashboardStore.getState().latestFatigue?.fatigueScore).toBe(70);
  });

  // ── Recommendations ──────────────────────────────────────────────────────

  it("adds a recommendation", () => {
    useDashboardStore.getState().addRecommendation(makeRec("r1"));
    expect(useDashboardStore.getState().recommendations).toHaveLength(1);
  });

  it("does not add duplicate recommendations", () => {
    useDashboardStore.getState().addRecommendation(makeRec("dup"));
    useDashboardStore.getState().addRecommendation(makeRec("dup"));
    expect(useDashboardStore.getState().recommendations).toHaveLength(1);
  });

  it("limits recommendations to 10", () => {
    for (let i = 0; i < 15; i++) {
      useDashboardStore.getState().addRecommendation(makeRec(`r${i}`));
    }
    expect(useDashboardStore.getState().recommendations.length).toBeLessThanOrEqual(10);
  });

  it("dismisses a recommendation", () => {
    useDashboardStore.getState().addRecommendation(makeRec("dismiss-me"));
    useDashboardStore.getState().dismissRecommendation("dismiss-me");
    expect(useDashboardStore.getState().recommendations).toHaveLength(0);
  });

  it("accepts a recommendation", () => {
    useDashboardStore.getState().addRecommendation(makeRec("accept-me"));
    useDashboardStore.getState().acceptRecommendation("accept-me");
    const rec = useDashboardStore.getState().recommendations.find(r => r.id === "accept-me");
    expect(rec?.accepted).toBe(true);
  });

  // ── Sensor toggles ───────────────────────────────────────────────────────

  it("toggles camera", () => {
    expect(useDashboardStore.getState().isCameraEnabled).toBe(false);
    useDashboardStore.getState().toggleCamera();
    expect(useDashboardStore.getState().isCameraEnabled).toBe(true);
    useDashboardStore.getState().toggleCamera();
    expect(useDashboardStore.getState().isCameraEnabled).toBe(false);
  });

  it("toggles microphone", () => {
    useDashboardStore.getState().toggleMic();
    expect(useDashboardStore.getState().isMicEnabled).toBe(true);
  });

  it("toggles keyboard tracking", () => {
    useDashboardStore.getState().toggleKeyboardTracking();
    expect(useDashboardStore.getState().isKeyboardTracking).toBe(true);
  });

  // ── Alerts ───────────────────────────────────────────────────────────────

  it("increments alert count", () => {
    useDashboardStore.getState().incrementAlert();
    useDashboardStore.getState().incrementAlert();
    expect(useDashboardStore.getState().alertCount).toBe(2);
  });

  it("clears alerts", () => {
    useDashboardStore.getState().incrementAlert();
    useDashboardStore.getState().clearAlerts();
    expect(useDashboardStore.getState().alertCount).toBe(0);
  });

  // ── Reset ────────────────────────────────────────────────────────────────

  it("resets all state", () => {
    useDashboardStore.getState().updateFatigue(makeFatigue(80));
    useDashboardStore.getState().addRecommendation(makeRec("r1"));
    useDashboardStore.getState().reset();
    expect(useDashboardStore.getState().dashboard.fatigueScore).toBe(0);
    expect(useDashboardStore.getState().recommendations).toHaveLength(0);
  });
});
