import { create } from "zustand";
import { immer } from "zustand/middleware/immer";
import type {
  DashboardState,
  FatigueMetrics,
  VoiceStressMetrics,
  BehavioralMetrics,
  ProductivityPrediction,
  Recommendation,
  WorkSession,
  TimeSeriesDataPoint,
} from "@/types";

// -----------------------------------------------------------
// History buffers (for sparklines / trend charts)
// -----------------------------------------------------------
const MAX_HISTORY = 60; // 60 data points

function appendToHistory(
  history: TimeSeriesDataPoint[],
  value: number
): TimeSeriesDataPoint[] {
  const next = [
    ...history,
    { timestamp: new Date().toISOString(), value },
  ];
  return next.length > MAX_HISTORY ? next.slice(-MAX_HISTORY) : next;
}

// -----------------------------------------------------------
// State Shape
// -----------------------------------------------------------
interface DashboardStoreState {
  // Current scores
  dashboard: DashboardState;

  // Latest detailed metrics
  latestFatigue: FatigueMetrics | null;
  latestStress: VoiceStressMetrics | null;
  latestBehavioral: BehavioralMetrics | null;
  latestPrediction: ProductivityPrediction | null;

  // Active recommendations
  recommendations: Recommendation[];

  // Historical data for charts
  fatigueHistory: TimeSeriesDataPoint[];
  stressHistory: TimeSeriesDataPoint[];
  productivityHistory: TimeSeriesDataPoint[];
  focusHistory: TimeSeriesDataPoint[];

  // UI state
  isMonitoringActive: boolean;
  isCameraEnabled: boolean;
  isMicEnabled: boolean;
  isKeyboardTracking: boolean;
  alertCount: number;
}

interface DashboardStoreActions {
  updateFatigue: (metrics: FatigueMetrics) => void;
  updateStress: (metrics: VoiceStressMetrics) => void;
  updateBehavioral: (metrics: BehavioralMetrics) => void;
  updatePrediction: (prediction: ProductivityPrediction) => void;
  addRecommendation: (rec: Recommendation) => void;
  dismissRecommendation: (id: string) => void;
  acceptRecommendation: (id: string) => void;
  startSession: (session: WorkSession) => void;
  endSession: () => void;
  toggleMonitoring: () => void;
  toggleCamera: () => void;
  toggleMic: () => void;
  toggleKeyboardTracking: () => void;
  incrementAlert: () => void;
  clearAlerts: () => void;
  reset: () => void;
}

type DashboardStore = DashboardStoreState & DashboardStoreActions;

// -----------------------------------------------------------
// Initial State
// -----------------------------------------------------------
const initialState: DashboardStoreState = {
  dashboard: {
    fatigueScore: 0,
    stressScore: 0,
    productivityScore: 85,
    focusLevel: 0,
    burnoutRisk: 0,
    currentStreak: 0,
    isMonitoring: false,
    activeSession: null,
  },
  latestFatigue: null,
  latestStress: null,
  latestBehavioral: null,
  latestPrediction: null,
  recommendations: [],
  fatigueHistory: [],
  stressHistory: [],
  productivityHistory: [],
  focusHistory: [],
  isMonitoringActive: false,
  isCameraEnabled: false,
  isMicEnabled: false,
  isKeyboardTracking: false,
  alertCount: 0,
};

// -----------------------------------------------------------
// Store
// -----------------------------------------------------------
export const useDashboardStore = create<DashboardStore>()(
  immer((set) => ({
    ...initialState,

    updateFatigue: (metrics: FatigueMetrics) => {
      set((state) => {
        state.latestFatigue = metrics;
        state.dashboard.fatigueScore = metrics.fatigueScore;
        state.fatigueHistory = appendToHistory(
          state.fatigueHistory,
          metrics.fatigueScore
        );
      });
    },

    updateStress: (metrics: VoiceStressMetrics) => {
      set((state) => {
        state.latestStress = metrics;
        state.dashboard.stressScore = metrics.stressScore;
        state.stressHistory = appendToHistory(
          state.stressHistory,
          metrics.stressScore
        );
      });
    },

    updateBehavioral: (metrics: BehavioralMetrics) => {
      set((state) => {
        state.latestBehavioral = metrics;
        state.dashboard.focusLevel = metrics.behaviorScore;
        state.focusHistory = appendToHistory(
          state.focusHistory,
          metrics.behaviorScore
        );
      });
    },

    updatePrediction: (prediction: ProductivityPrediction) => {
      set((state) => {
        state.latestPrediction = prediction;
        state.dashboard.productivityScore = prediction.productivityScore;
        state.dashboard.burnoutRisk = Math.round(prediction.burnoutProbability * 100);
        state.productivityHistory = appendToHistory(
          state.productivityHistory,
          prediction.productivityScore
        );
      });
    },

    addRecommendation: (rec: Recommendation) => {
      set((state) => {
        // Avoid duplicates and keep max 10
        const exists = state.recommendations.some((r) => r.id === rec.id);
        if (!exists) {
          state.recommendations.unshift(rec);
          if (state.recommendations.length > 10) {
            state.recommendations = state.recommendations.slice(0, 10);
          }
        }
      });
    },

    dismissRecommendation: (id: string) => {
      set((state) => {
        state.recommendations = state.recommendations.filter((r) => r.id !== id);
      });
    },

    acceptRecommendation: (id: string) => {
      set((state) => {
        const rec = state.recommendations.find((r) => r.id === id);
        if (rec) rec.accepted = true;
      });
    },

    startSession: (session: WorkSession) => {
      set((state) => {
        state.dashboard.activeSession = session;
        state.dashboard.isMonitoring = true;
        state.isMonitoringActive = true;
      });
    },

    endSession: () => {
      set((state) => {
        state.dashboard.activeSession = null;
        state.dashboard.isMonitoring = false;
        state.isMonitoringActive = false;
      });
    },

    toggleMonitoring: () => {
      set((state) => {
        state.isMonitoringActive = !state.isMonitoringActive;
        state.dashboard.isMonitoring = state.isMonitoringActive;
      });
    },

    toggleCamera: () => {
      set((state) => {
        state.isCameraEnabled = !state.isCameraEnabled;
      });
    },

    toggleMic: () => {
      set((state) => {
        state.isMicEnabled = !state.isMicEnabled;
      });
    },

    toggleKeyboardTracking: () => {
      set((state) => {
        state.isKeyboardTracking = !state.isKeyboardTracking;
      });
    },

    incrementAlert: () => {
      set((state) => {
        state.alertCount += 1;
      });
    },

    clearAlerts: () => {
      set((state) => {
        state.alertCount = 0;
      });
    },

    reset: () => {
      set(() => ({ ...initialState }));
    },
  }))
);
