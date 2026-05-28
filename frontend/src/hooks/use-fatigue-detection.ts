"use client";

import { useRef, useEffect, useCallback, useState } from "react";
import type { FatigueMetrics, DrowsinessLevel } from "@/types";
import { useDashboardStore } from "@/store/dashboard-store";
import { apiClient } from "@/lib/api-client";

// -----------------------------------------------------------
// EAR/MAR Calculation Utilities
// -----------------------------------------------------------

/**
 * Eye Aspect Ratio (EAR) — Drowsiness detection formula.
 * EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
 *
 * Returns a value between 0 (fully closed) and ~0.4 (wide open).
 * Typical drowsiness threshold: EAR < 0.25
 */
function calculateEAR(landmarks: { x: number; y: number }[]): number {
  const [p1, p2, p3, p4, p5, p6] = landmarks;
  const vertical1 = euclidean(p2, p6);
  const vertical2 = euclidean(p3, p5);
  const horizontal = euclidean(p1, p4);
  if (horizontal === 0) return 0;
  return (vertical1 + vertical2) / (2 * horizontal);
}

/**
 * Mouth Aspect Ratio (MAR) — Yawn detection.
 * MAR > 0.7 typically indicates yawning.
 */
function calculateMAR(landmarks: { x: number; y: number }[]): number {
  const [p1, p2, p3, p4, p5, p6] = landmarks;
  const vertical1 = euclidean(p2, p6);
  const vertical2 = euclidean(p3, p5);
  const horizontal = euclidean(p1, p4);
  if (horizontal === 0) return 0;
  return (vertical1 + vertical2) / (2 * horizontal);
}

function euclidean(a: { x: number; y: number }, b: { x: number; y: number }): number {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

function getDrowsinessLevel(fatigueScore: number): DrowsinessLevel {
  if (fatigueScore < 20) return "alert";
  if (fatigueScore < 40) return "mild";
  if (fatigueScore < 60) return "moderate";
  if (fatigueScore < 80) return "severe";
  return "critical";
}

// -----------------------------------------------------------
// Blink Counter
// -----------------------------------------------------------
class BlinkCounter {
  private blinkCount = 0;
  private windowStart = Date.now();
  private wasEyeOpen = true;
  private readonly windowDuration = 60_000; // 1 minute

  update(ear: number, threshold = 0.25): void {
    const isEyeClosed = ear < threshold;
    if (isEyeClosed && this.wasEyeOpen) {
      this.blinkCount++;
    }
    this.wasEyeOpen = !isEyeClosed;
  }

  getBlinkRate(): number {
    const now = Date.now();
    const elapsed = now - this.windowStart;
    if (elapsed >= this.windowDuration) {
      const rate = (this.blinkCount / elapsed) * 60_000;
      this.blinkCount = 0;
      this.windowStart = now;
      return rate;
    }
    // Extrapolate
    return elapsed > 0 ? (this.blinkCount / elapsed) * 60_000 : 0;
  }
}

// -----------------------------------------------------------
// Rolling Fatigue Score
// -----------------------------------------------------------
class FatigueScoreCalculator {
  private earHistory: number[] = [];
  private marHistory: number[] = [];
  private readonly windowSize = 30;

  update(ear: number, mar: number): number {
    this.earHistory.push(ear);
    this.marHistory.push(mar);
    if (this.earHistory.length > this.windowSize) this.earHistory.shift();
    if (this.marHistory.length > this.windowSize) this.marHistory.shift();

    const avgEar = this.earHistory.reduce((a, b) => a + b, 0) / this.earHistory.length;
    const avgMar = this.marHistory.reduce((a, b) => a + b, 0) / this.marHistory.length;

    // Normalize EAR (0.4 = fully alert, 0 = closed)
    const earScore = Math.max(0, Math.min(1, 1 - avgEar / 0.4));

    // Yawn contributes to fatigue
    const yawnScore = Math.max(0, Math.min(1, avgMar / 0.8));

    // Weighted combination
    return Math.round((earScore * 0.7 + yawnScore * 0.3) * 100);
  }
}

// -----------------------------------------------------------
// Hook
// -----------------------------------------------------------
interface UseFatigueDetectionOptions {
  sessionId: string;
  userId: string;
  enabled: boolean;
  onFatigueUpdate?: (metrics: FatigueMetrics) => void;
  onCriticalAlert?: () => void;
}

export function useFatigueDetection({
  sessionId,
  userId,
  enabled,
  onFatigueUpdate,
  onCriticalAlert,
}: UseFatigueDetectionOptions) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef<number>(0);
  const streamRef = useRef<MediaStream | null>(null);
  const blinkCounter = useRef(new BlinkCounter());
  const fatigueCalc = useRef(new FatigueScoreCalculator());
  const lastSendRef = useRef(0);
  const SEND_INTERVAL = 2000; // Send to backend every 2s

  const [isInitialized, setIsInitialized] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const updateFatigue = useDashboardStore((s) => s.updateFatigue);

  // -----------------------------------------------------------
  // MediaPipe Face Mesh initialization (lazy load)
  // -----------------------------------------------------------
  const initMediaPipe = useCallback(async () => {
    try {
      // Dynamic import — MediaPipe is large, only load when needed
      const { FaceMesh } = await import("@mediapipe/face_mesh");
      const { Camera } = await import("@mediapipe/camera_utils");

      if (!videoRef.current) return;

      const faceMesh = new FaceMesh({
        locateFile: (file: string) =>
          `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
      });

      faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.7,
        minTrackingConfidence: 0.7,
      });

      faceMesh.onResults((results: { multiFaceLandmarks?: Array<Array<{ x: number; y: number; z: number }>> }) => {
        if (!results.multiFaceLandmarks?.length) return;

        const landmarks = results.multiFaceLandmarks[0];

        // MediaPipe Face Mesh eye landmark indices
        // Left eye: 33, 160, 158, 133, 153, 144
        // Right eye: 362, 385, 387, 263, 373, 380
        const leftEye = [33, 160, 158, 133, 153, 144].map((i) => landmarks[i]);
        const rightEye = [362, 385, 387, 263, 373, 380].map((i) => landmarks[i]);

        // Mouth landmarks: 61, 291, 39, 181, 0, 17
        const mouth = [61, 291, 39, 181, 0, 17].map((i) => landmarks[i]);

        const leftEAR = calculateEAR(leftEye);
        const rightEAR = calculateEAR(rightEye);
        const ear = (leftEAR + rightEAR) / 2;
        const mar = calculateMAR(mouth);

        blinkCounter.current.update(ear);
        const fatigueScore = fatigueCalc.current.update(ear, mar);
        const blinkRate = blinkCounter.current.getBlinkRate();

        const metrics: FatigueMetrics = {
          id: crypto.randomUUID(),
          userId,
          sessionId,
          timestamp: new Date().toISOString(),
          blinkRate,
          eyeAspectRatio: ear,
          mouthAspectRatio: mar,
          headTiltAngle: 0, // TODO: head pose estimation
          gazeDrift: 0,
          fatigueScore,
          drowsinessLevel: getDrowsinessLevel(fatigueScore),
          confidence: 0.9,
        };

        updateFatigue(metrics);
        onFatigueUpdate?.(metrics);

        if (fatigueScore >= 80) {
          onCriticalAlert?.();
        }

        // Send to backend every SEND_INTERVAL ms
        const now = Date.now();
        if (now - lastSendRef.current > SEND_INTERVAL) {
          lastSendRef.current = now;
          apiClient
            .post("/fatigue/metrics", metrics)
            .catch(console.error);
        }
      });

      const camera = new Camera(videoRef.current, {
        onFrame: async () => {
          if (videoRef.current) {
            await faceMesh.send({ image: videoRef.current });
          }
        },
        width: 640,
        height: 480,
      });

      await camera.start();
      setIsInitialized(true);
    } catch (err) {
      setError(`Failed to initialize face detection: ${(err as Error).message}`);
    }
  }, [userId, sessionId, updateFatigue, onFatigueUpdate, onCriticalAlert]);

  // -----------------------------------------------------------
  // Camera permission & stream
  // -----------------------------------------------------------
  useEffect(() => {
    if (!enabled) return;

    let mounted = true;

    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: "user",
            frameRate: { ideal: 15, max: 30 },
          },
        });

        if (!mounted) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }

        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current?.play();
            initMediaPipe();
          };
        }
      } catch (err) {
        if ((err as Error).name === "NotAllowedError") {
          setError("Camera permission denied. Please allow camera access.");
        } else {
          setError(`Camera error: ${(err as Error).message}`);
        }
      }
    };

    startCamera();

    return () => {
      mounted = false;
      cancelAnimationFrame(animFrameRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
      setIsInitialized(false);
    };
  }, [enabled, initMediaPipe]);

  return {
    videoRef,
    canvasRef,
    isInitialized,
    error,
  };
}
