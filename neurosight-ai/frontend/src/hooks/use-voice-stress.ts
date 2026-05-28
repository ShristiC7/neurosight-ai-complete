"use client";

import { useRef, useEffect, useCallback, useState } from "react";
import type { VoiceStressMetrics, EmotionState } from "@/types";
import { useDashboardStore } from "@/store/dashboard-store";
import { apiClient } from "@/lib/api-client";

// -----------------------------------------------------------
// MFCC Extraction (simplified browser-side)
// Full MFCC runs server-side via librosa — this is a lightweight
// approximation for real-time visual feedback.
// -----------------------------------------------------------

function hammingWindow(size: number): Float32Array {
  const window = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    window[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (size - 1));
  }
  return window;
}

function computeZCR(buffer: Float32Array): number {
  let count = 0;
  for (let i = 1; i < buffer.length; i++) {
    if ((buffer[i] >= 0) !== (buffer[i - 1] >= 0)) count++;
  }
  return count / buffer.length;
}

function computeRMS(buffer: Float32Array): number {
  const sumSq = buffer.reduce((acc, val) => acc + val * val, 0);
  return Math.sqrt(sumSq / buffer.length);
}

// Spectral centroid from FFT magnitudes
function computeSpectralCentroid(magnitudes: Uint8Array, sampleRate: number): number {
  let weightedSum = 0;
  let totalMagnitude = 0;
  const binWidth = sampleRate / (2 * magnitudes.length);

  for (let i = 0; i < magnitudes.length; i++) {
    const freq = i * binWidth;
    weightedSum += freq * magnitudes[i];
    totalMagnitude += magnitudes[i];
  }

  return totalMagnitude > 0 ? weightedSum / totalMagnitude : 0;
}

// Map audio features to stress score (0-100)
function computeStressScore(features: {
  zcr: number;
  rms: number;
  spectralCentroid: number;
  pitchVariance: number;
}): number {
  // High ZCR + high energy + high spectral centroid = more stressed
  const normalizedZCR = Math.min(features.zcr * 10, 1);
  const normalizedRMS = Math.min(features.rms * 5, 1);
  const normalizedSC = Math.min(features.spectralCentroid / 4000, 1);
  const normalizedPitch = Math.min(features.pitchVariance / 100, 1);

  return Math.round(
    (normalizedZCR * 0.25 + normalizedRMS * 0.3 + normalizedSC * 0.25 + normalizedPitch * 0.2) * 100
  );
}

function getEmotionState(stressScore: number): EmotionState {
  if (stressScore < 20) return "calm";
  if (stressScore < 40) return "energetic";
  if (stressScore < 60) return "stressed";
  if (stressScore < 80) return "fatigued";
  return "anxious";
}

// -----------------------------------------------------------
// Hook
// -----------------------------------------------------------
interface UseVoiceStressOptions {
  sessionId: string;
  userId: string;
  enabled: boolean;
  onStressUpdate?: (metrics: VoiceStressMetrics) => void;
  onHighStress?: () => void;
}

const ANALYSIS_INTERVAL = 3000; // Analyze every 3 seconds
const FFT_SIZE = 2048;
const SEND_INTERVAL = 5000;

export function useVoiceStress({
  sessionId,
  userId,
  enabled,
  onStressUpdate,
  onHighStress,
}: UseVoiceStressOptions) {
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastSendRef = useRef(0);
  const pitchHistoryRef = useRef<number[]>([]);

  const [isListening, setIsListening] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [volumeLevel, setVolumeLevel] = useState(0);

  const updateStress = useDashboardStore((s) => s.updateStress);

  const analyze = useCallback(() => {
    if (!analyserRef.current) return;

    const analyser = analyserRef.current;
    const bufferLength = analyser.frequencyBinCount;
    const freqData = new Uint8Array(bufferLength);
    const timeData = new Float32Array(analyser.fftSize);

    analyser.getByteFrequencyData(freqData);
    analyser.getFloatTimeDomainData(timeData);

    const zcr = computeZCR(timeData);
    const rms = computeRMS(timeData);
    const spectralCentroid = computeSpectralCentroid(
      freqData,
      audioContextRef.current!.sampleRate
    );

    // Simple pitch estimation via autocorrelation
    const pitchEstimate = spectralCentroid / 2;
    pitchHistoryRef.current.push(pitchEstimate);
    if (pitchHistoryRef.current.length > 20) pitchHistoryRef.current.shift();

    const pitchMean =
      pitchHistoryRef.current.reduce((a, b) => a + b, 0) /
      pitchHistoryRef.current.length;
    const pitchVariance =
      pitchHistoryRef.current.reduce((acc, v) => acc + (v - pitchMean) ** 2, 0) /
      pitchHistoryRef.current.length;

    // Volume level for UI visualization (0-1)
    const normalizedVolume = Math.min(rms * 3, 1);
    setVolumeLevel(normalizedVolume);

    const stressScore = computeStressScore({ zcr, rms, spectralCentroid, pitchVariance });

    const metrics: VoiceStressMetrics = {
      id: crypto.randomUUID(),
      userId,
      sessionId,
      timestamp: new Date().toISOString(),
      pitchVariance,
      speechEnergy: rms,
      pauseDuration: 0, // Requires longer-term analysis
      stressScore,
      emotionState: getEmotionState(stressScore),
      mfccFeatures: Array.from(freqData.slice(0, 13)).map((v) => v / 255),
      confidence: 0.75,
    };

    updateStress(metrics);
    onStressUpdate?.(metrics);

    if (stressScore >= 70) {
      onHighStress?.();
    }

    // Send to backend for full librosa analysis
    const now = Date.now();
    if (now - lastSendRef.current > SEND_INTERVAL) {
      lastSendRef.current = now;

      // Package audio data as Float32Array for backend
      const audioData = Array.from(timeData);
      apiClient
        .post("/audio/analyze", {
          userId,
          sessionId,
          audioData,
          sampleRate: audioContextRef.current!.sampleRate,
          precomputed: { zcr, rms, spectralCentroid },
        })
        .catch(console.error);
    }
  }, [userId, sessionId, updateStress, onStressUpdate, onHighStress]);

  useEffect(() => {
    if (!enabled) return;

    let mounted = true;

    const startAudio = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            sampleRate: 22050, // Standard for audio ML
            channelCount: 1,
          },
        });

        if (!mounted) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }

        streamRef.current = stream;

        const ctx = new AudioContext({ sampleRate: 22050 });
        audioContextRef.current = ctx;

        const source = ctx.createMediaStreamSource(stream);
        const analyser = ctx.createAnalyser();
        analyser.fftSize = FFT_SIZE;
        analyser.smoothingTimeConstant = 0.8;

        source.connect(analyser);
        analyserRef.current = analyser;

        setIsListening(true);

        // Start analysis loop
        intervalRef.current = setInterval(analyze, ANALYSIS_INTERVAL);
      } catch (err) {
        if ((err as Error).name === "NotAllowedError") {
          setError("Microphone permission denied.");
        } else {
          setError(`Audio error: ${(err as Error).message}`);
        }
      }
    };

    startAudio();

    return () => {
      mounted = false;
      if (intervalRef.current) clearInterval(intervalRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
      audioContextRef.current?.close();
      setIsListening(false);
    };
  }, [enabled, analyze]);

  return { isListening, error, volumeLevel };
}
