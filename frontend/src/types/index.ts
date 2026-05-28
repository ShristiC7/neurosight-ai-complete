// ============================================================
// NeuroSight AI — Core Type Definitions
// ============================================================

// -----------------------------------------------------------
// User & Auth Types
// -----------------------------------------------------------
export interface User {
  id: string;
  name: string;
  email: string;
  avatarUrl?: string;
  createdAt: string;
  preferences: UserPreferences;
}

export interface UserPreferences {
  workHoursStart: number; // 0-23
  workHoursEnd: number;
  breakDuration: number; // minutes
  timezone: string;
  theme: "dark" | "light" | "system";
  notifications: NotificationSettings;
}

export interface NotificationSettings {
  fatigueAlerts: boolean;
  breakReminders: boolean;
  productivityInsights: boolean;
  burnoutWarnings: boolean;
}

export interface AuthSession {
  user: User;
  accessToken: string;
  expiresAt: number;
}

// -----------------------------------------------------------
// Fatigue Detection Types
// -----------------------------------------------------------
export interface FatigueMetrics {
  id: string;
  userId: string;
  sessionId: string;
  timestamp: string;
  blinkRate: number; // blinks per minute
  eyeAspectRatio: number; // EAR value
  mouthAspectRatio: number; // MAR value
  headTiltAngle: number; // degrees
  gazeDrift: number; // 0-1 normalized
  fatigueScore: number; // 0-100
  drowsinessLevel: DrowsinessLevel;
  confidence: number; // 0-1
}

export type DrowsinessLevel = "alert" | "mild" | "moderate" | "severe" | "critical";

export interface EyeLandmark {
  x: number;
  y: number;
  z: number;
}

export interface FatigueFrame {
  frameId: string;
  timestamp: number;
  landmarks: EyeLandmark[];
  ear: number;
  mar: number;
  fatigueScore: number;
}

// -----------------------------------------------------------
// Voice Stress Types
// -----------------------------------------------------------
export interface VoiceStressMetrics {
  id: string;
  userId: string;
  sessionId: string;
  timestamp: string;
  pitchVariance: number;
  speechEnergy: number;
  pauseDuration: number; // milliseconds
  stressScore: number; // 0-100
  emotionState: EmotionState;
  mfccFeatures: number[]; // 13-dimensional MFCC vector
  confidence: number;
}

export type EmotionState = "calm" | "stressed" | "fatigued" | "energetic" | "anxious";

export interface AudioFeatures {
  mfcc: number[];
  chroma: number[];
  spectralCentroid: number;
  spectralContrast: number[];
  zeroCrossingRate: number;
  rmsEnergy: number;
}

// -----------------------------------------------------------
// Behavioral Analytics Types
// -----------------------------------------------------------
export interface BehavioralMetrics {
  id: string;
  userId: string;
  sessionId: string;
  timestamp: string;
  typingSpeed: number; // WPM
  typingRhythmVariance: number;
  errorRate: number; // errors per minute
  mouseMovementEntropy: number; // 0-1
  mouseClickRate: number;
  appSwitchFrequency: number; // switches per hour
  focusSessionDuration: number; // minutes
  idleTime: number; // seconds
  behaviorScore: number; // 0-100 (100 = normal)
  anomalyScore: number; // 0-1 (1 = highly anomalous)
}

export interface KeystrokeData {
  key: string;
  dwellTime: number; // ms key held down
  flightTime: number; // ms between keystrokes
  timestamp: number;
}

export interface MouseEvent {
  x: number;
  y: number;
  velocity: number;
  timestamp: number;
  eventType: "move" | "click" | "scroll";
}

// -----------------------------------------------------------
// Productivity Prediction Types
// -----------------------------------------------------------
export interface ProductivityPrediction {
  id: string;
  userId: string;
  timestamp: string;
  productivityScore: number; // 0-100
  burnoutProbability: number; // 0-1
  focusWindowStart: string; // ISO timestamp
  focusWindowEnd: string;
  cognitiveLoad: number; // 0-100
  recommendedBreakAt: string; // ISO timestamp
  predictedCrashAt?: string; // ISO timestamp
  confidence: number;
}

export interface FocusWindow {
  start: string;
  end: string;
  quality: "peak" | "good" | "moderate" | "poor";
  score: number;
}

export interface TimeSeriesDataPoint {
  timestamp: string;
  value: number;
  label?: string;
}

// -----------------------------------------------------------
// Recommendation Types
// -----------------------------------------------------------
export interface Recommendation {
  id: string;
  userId: string;
  sessionId: string;
  timestamp: string;
  type: RecommendationType;
  priority: RecommendationPriority;
  title: string;
  message: string;
  actionLabel?: string;
  actionUrl?: string;
  durationMinutes?: number;
  accepted: boolean | null;
  expiresAt: string;
  metadata: Record<string, unknown>;
}

export type RecommendationType =
  | "take_break"
  | "stretch"
  | "hydrate"
  | "deep_work"
  | "light_task"
  | "sleep"
  | "exercise"
  | "meditation"
  | "eye_rest"
  | "posture_check";

export type RecommendationPriority = "critical" | "high" | "medium" | "low";

// -----------------------------------------------------------
// Session Types
// -----------------------------------------------------------
export interface WorkSession {
  id: string;
  userId: string;
  startTime: string;
  endTime?: string;
  isActive: boolean;
  avgFatigueScore: number;
  avgProductivityScore: number;
  avgStressScore: number;
  totalFocusTime: number; // minutes
  breaksTaken: number;
  totalKeystrokes: number;
}

// -----------------------------------------------------------
// Dashboard / Realtime Types
// -----------------------------------------------------------
export interface DashboardState {
  fatigueScore: number;
  stressScore: number;
  productivityScore: number;
  focusLevel: number;
  burnoutRisk: number;
  currentStreak: number; // minutes in focus
  isMonitoring: boolean;
  activeSession: WorkSession | null;
}

export interface RealtimeUpdate {
  type: "fatigue" | "stress" | "behavioral" | "prediction" | "recommendation";
  timestamp: string;
  data: FatigueMetrics | VoiceStressMetrics | BehavioralMetrics | ProductivityPrediction | Recommendation;
}

// -----------------------------------------------------------
// API Response Types
// -----------------------------------------------------------
export interface ApiResponse<T> {
  data: T;
  status: "success" | "error";
  message?: string;
  timestamp: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  totalPages: number;
}

export interface ApiError {
  statusCode: number;
  message: string;
  details?: Record<string, string[]>;
  timestamp: string;
}

// -----------------------------------------------------------
// WebSocket Message Types
// -----------------------------------------------------------
export interface WSMessage<T = unknown> {
  event: string;
  payload: T;
  sessionId: string;
  timestamp: number;
}

export type WSEvent =
  | "fatigue:update"
  | "stress:update"
  | "behavioral:update"
  | "prediction:update"
  | "recommendation:new"
  | "session:start"
  | "session:end"
  | "alert:critical";

// -----------------------------------------------------------
// Chart / Visualization Types
// -----------------------------------------------------------
export interface ChartDataPoint {
  x: string | number;
  y: number;
  category?: string;
  metadata?: Record<string, unknown>;
}

export interface HeatmapCell {
  hour: number;
  day: number;
  value: number;
  label: string;
}
