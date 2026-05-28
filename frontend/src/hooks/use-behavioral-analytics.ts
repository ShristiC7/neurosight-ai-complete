"use client";

import { useEffect, useRef, useCallback } from "react";
import type { BehavioralMetrics, KeystrokeData, MouseEvent as NSMouseEvent } from "@/types";
import { useDashboardStore } from "@/store/dashboard-store";
import { apiClient } from "@/lib/api-client";

// -----------------------------------------------------------
// Shannon Entropy — measures randomness of mouse movement
// -----------------------------------------------------------
function shannonEntropy(values: number[]): number {
  if (values.length === 0) return 0;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;

  // Bin into 10 buckets
  const bins = new Array(10).fill(0);
  values.forEach((v) => {
    const idx = Math.min(9, Math.floor(((v - min) / range) * 10));
    bins[idx]++;
  });

  const total = values.length;
  return -bins.reduce((acc, count) => {
    if (count === 0) return acc;
    const p = count / total;
    return acc + p * Math.log2(p);
  }, 0);
}

// -----------------------------------------------------------
// Typing WPM Calculator
// -----------------------------------------------------------
class TypingAnalyzer {
  private keystrokes: KeystrokeData[] = [];
  private windowDuration = 60_000; // 1 minute window

  recordKeystroke(data: KeystrokeData): void {
    this.keystrokes.push(data);
    // Trim old keystrokes
    const cutoff = Date.now() - this.windowDuration;
    this.keystrokes = this.keystrokes.filter((k) => k.timestamp > cutoff);
  }

  getWPM(): number {
    // Average word = 5 characters
    const charsPerMinute = this.keystrokes.length;
    return Math.round(charsPerMinute / 5);
  }

  getFlightTimeVariance(): number {
    if (this.keystrokes.length < 2) return 0;
    const flightTimes = this.keystrokes
      .slice(1)
      .map((k) => k.flightTime)
      .filter((t) => t > 0 && t < 2000);

    if (flightTimes.length === 0) return 0;
    const mean = flightTimes.reduce((a, b) => a + b, 0) / flightTimes.length;
    const variance =
      flightTimes.reduce((acc, t) => acc + (t - mean) ** 2, 0) / flightTimes.length;
    return Math.sqrt(variance); // Standard deviation
  }

  getErrorRate(): number {
    const backspaces = this.keystrokes.filter(
      (k) => k.key === "Backspace" || k.key === "Delete"
    ).length;
    return this.keystrokes.length > 0
      ? (backspaces / this.keystrokes.length) * 60
      : 0;
  }
}

// -----------------------------------------------------------
// Mouse Analyzer
// -----------------------------------------------------------
class MouseAnalyzer {
  private events: NSMouseEvent[] = [];
  private readonly windowDuration = 30_000; // 30 seconds

  recordEvent(event: NSMouseEvent): void {
    this.events.push(event);
    const cutoff = Date.now() - this.windowDuration;
    this.events = this.events.filter((e) => e.timestamp > cutoff);
  }

  getEntropy(): number {
    if (this.events.length < 2) return 0;
    const velocities = this.events.map((e) => e.velocity).filter((v) => v > 0);
    return shannonEntropy(velocities) / 3.32; // Normalize by log2(10)
  }

  getClickRate(): number {
    const clicks = this.events.filter((e) => e.eventType === "click");
    return (clicks.length / 30) * 60; // per minute
  }
}

// -----------------------------------------------------------
// App Switch Tracker
// -----------------------------------------------------------
class AppSwitchTracker {
  private switches: number[] = [];
  private lastActiveAt = Date.now();

  recordSwitch(): void {
    this.switches.push(Date.now());
    const hourAgo = Date.now() - 3_600_000;
    this.switches = this.switches.filter((t) => t > hourAgo);
  }

  getSwitchFrequency(): number {
    return this.switches.length; // per hour
  }

  getFocusDuration(): number {
    return Math.round((Date.now() - this.lastActiveAt) / 60_000);
  }
}

// -----------------------------------------------------------
// Hook
// -----------------------------------------------------------
interface UseBehavioralAnalyticsOptions {
  sessionId: string;
  userId: string;
  enabled: boolean;
  onBehavioralUpdate?: (metrics: BehavioralMetrics) => void;
}

const AGGREGATE_INTERVAL = 10_000; // Aggregate every 10 seconds
const SEND_INTERVAL = 30_000; // Send to backend every 30 seconds

export function useBehavioralAnalytics({
  sessionId,
  userId,
  enabled,
  onBehavioralUpdate,
}: UseBehavioralAnalyticsOptions) {
  const typingAnalyzer = useRef(new TypingAnalyzer());
  const mouseAnalyzer = useRef(new MouseAnalyzer());
  const appSwitchTracker = useRef(new AppSwitchTracker());
  const lastKeyTimestamp = useRef(0);
  const aggregateIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastSendRef = useRef(0);

  const updateBehavioral = useDashboardStore((s) => s.updateBehavioral);

  const computeBehaviorScore = (
    wpm: number,
    errorRate: number,
    entropy: number
  ): number => {
    // Normalize against baseline
    const wpmScore = Math.min(100, (wpm / 60) * 100);
    const errorScore = Math.max(0, 100 - errorRate * 20);
    const entropyScore = Math.min(100, entropy * 100);
    return Math.round(wpmScore * 0.4 + errorScore * 0.4 + entropyScore * 0.2);
  };

  const aggregate = useCallback(() => {
    const wpm = typingAnalyzer.current.getWPM();
    const rhythmVariance = typingAnalyzer.current.getFlightTimeVariance();
    const errorRate = typingAnalyzer.current.getErrorRate();
    const entropy = mouseAnalyzer.current.getEntropy();
    const clickRate = mouseAnalyzer.current.getClickRate();
    const switchFreq = appSwitchTracker.current.getSwitchFrequency();
    const focusDuration = appSwitchTracker.current.getFocusDuration();
    const behaviorScore = computeBehaviorScore(wpm, errorRate, entropy);

    const metrics: BehavioralMetrics = {
      id: crypto.randomUUID(),
      userId,
      sessionId,
      timestamp: new Date().toISOString(),
      typingSpeed: wpm,
      typingRhythmVariance: rhythmVariance,
      errorRate,
      mouseMovementEntropy: entropy,
      mouseClickRate: clickRate,
      appSwitchFrequency: switchFreq,
      focusSessionDuration: focusDuration,
      idleTime: 0,
      behaviorScore,
      anomalyScore: Math.max(0, 1 - behaviorScore / 100),
    };

    updateBehavioral(metrics);
    onBehavioralUpdate?.(metrics);

    const now = Date.now();
    if (now - lastSendRef.current > SEND_INTERVAL) {
      lastSendRef.current = now;
      apiClient.post("/behavioral/metrics", metrics).catch(console.error);
    }
  }, [userId, sessionId, updateBehavioral, onBehavioralUpdate]);

  useEffect(() => {
    if (!enabled) return;

    // Keyboard tracking
    const handleKeyDown = (e: KeyboardEvent) => {
      const now = Date.now();
      const flightTime = lastKeyTimestamp.current
        ? now - lastKeyTimestamp.current
        : 0;

      typingAnalyzer.current.recordKeystroke({
        key: e.key,
        dwellTime: 0,
        flightTime,
        timestamp: now,
      });

      lastKeyTimestamp.current = now;
    };

    // Mouse tracking (throttled)
    let lastMouseEvent = 0;
    const handleMouseMove = (e: globalThis.MouseEvent) => {
      const now = Date.now();
      if (now - lastMouseEvent < 50) return; // 20fps max
      lastMouseEvent = now;

      mouseAnalyzer.current.recordEvent({
        x: e.clientX,
        y: e.clientY,
        velocity: 0, // Calculated on server
        timestamp: now,
        eventType: "move",
      });
    };

    const handleMouseClick = (e: globalThis.MouseEvent) => {
      mouseAnalyzer.current.recordEvent({
        x: e.clientX,
        y: e.clientY,
        velocity: 0,
        timestamp: Date.now(),
        eventType: "click",
      });
    };

    // Visibility change — app switching proxy
    const handleVisibilityChange = () => {
      if (document.hidden) {
        appSwitchTracker.current.recordSwitch();
      }
    };

    document.addEventListener("keydown", handleKeyDown, { passive: true });
    document.addEventListener("mousemove", handleMouseMove, { passive: true });
    document.addEventListener("click", handleMouseClick, { passive: true });
    document.addEventListener("visibilitychange", handleVisibilityChange);

    // Aggregate on interval
    aggregateIntervalRef.current = setInterval(aggregate, AGGREGATE_INTERVAL);

    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("click", handleMouseClick);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      if (aggregateIntervalRef.current) {
        clearInterval(aggregateIntervalRef.current);
      }
    };
  }, [enabled, aggregate]);
}
