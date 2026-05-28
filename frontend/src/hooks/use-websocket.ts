"use client";

import {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
  useCallback,
  type ReactNode,
} from "react";
import type { WSMessage, WSEvent, RealtimeUpdate } from "@/types";
import { useAuthStore } from "@/store/auth-store";

// -----------------------------------------------------------
// WebSocket Context
// -----------------------------------------------------------
interface WSContextValue {
  isConnected: boolean;
  isReconnecting: boolean;
  lastUpdate: RealtimeUpdate | null;
  send: (event: string, payload: unknown) => void;
  subscribe: (event: WSEvent, handler: (data: unknown) => void) => () => void;
}

const WSContext = createContext<WSContextValue | null>(null);

// -----------------------------------------------------------
// Provider
// -----------------------------------------------------------
const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000/ws";
const RECONNECT_INTERVALS = [1000, 2000, 5000, 10000, 30000];
const HEARTBEAT_INTERVAL = 25000;

export function WebSocketProvider({ children }: { children: ReactNode }) {
  const { accessToken, user } = useAuthStore();
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptRef = useRef(0);
  const heartbeatRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const listenersRef = useRef<Map<string, Set<(data: unknown) => void>>>(new Map());

  const [isConnected, setIsConnected] = useState(false);
  const [isReconnecting, setIsReconnecting] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<RealtimeUpdate | null>(null);

  const clearHeartbeat = useCallback(() => {
    if (heartbeatRef.current) {
      clearInterval(heartbeatRef.current);
      heartbeatRef.current = null;
    }
  }, []);

  const startHeartbeat = useCallback(() => {
    clearHeartbeat();
    heartbeatRef.current = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ event: "ping", payload: null }));
      }
    }, HEARTBEAT_INTERVAL);
  }, [clearHeartbeat]);

  const connect = useCallback(() => {
    if (!accessToken || !user) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const url = `${WS_URL}?token=${accessToken}&userId=${user.id}`;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      setIsReconnecting(false);
      reconnectAttemptRef.current = 0;
      startHeartbeat();
      console.info("[WS] Connected to NeuroSight stream");
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data) as WSMessage<unknown>;

        if (message.event === "pong") return;

        // Dispatch to all registered listeners for this event
        const handlers = listenersRef.current.get(message.event);
        handlers?.forEach((handler) => handler(message.payload));

        // Update last realtime update
        if (message.event.includes(":")) {
          setLastUpdate(message.payload as RealtimeUpdate);
        }
      } catch (err) {
        console.error("[WS] Failed to parse message:", err);
      }
    };

    ws.onclose = (event) => {
      setIsConnected(false);
      clearHeartbeat();

      if (event.code !== 1000) {
        // Abnormal close — schedule reconnect
        const attempt = reconnectAttemptRef.current;
        const delay = RECONNECT_INTERVALS[Math.min(attempt, RECONNECT_INTERVALS.length - 1)];
        setIsReconnecting(true);
        reconnectAttemptRef.current += 1;

        console.warn(`[WS] Disconnected. Reconnecting in ${delay}ms (attempt ${attempt + 1})`);
        setTimeout(connect, delay);
      }
    };

    ws.onerror = (err) => {
      console.error("[WS] Error:", err);
    };
  }, [accessToken, user, startHeartbeat, clearHeartbeat]);

  useEffect(() => {
    connect();
    return () => {
      clearHeartbeat();
      wsRef.current?.close(1000, "Component unmounted");
    };
  }, [connect, clearHeartbeat]);

  const send = useCallback((event: string, payload: unknown) => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) {
      console.warn("[WS] Cannot send — not connected");
      return;
    }
    const message: Partial<WSMessage> = {
      event,
      payload,
      timestamp: Date.now(),
    };
    wsRef.current.send(JSON.stringify(message));
  }, []);

  const subscribe = useCallback((event: WSEvent, handler: (data: unknown) => void) => {
    if (!listenersRef.current.has(event)) {
      listenersRef.current.set(event, new Set());
    }
    listenersRef.current.get(event)!.add(handler);

    // Return unsubscribe function
    return () => {
      listenersRef.current.get(event)?.delete(handler);
    };
  }, []);

  return (
    <WSContext.Provider value={{ isConnected, isReconnecting, lastUpdate, send, subscribe }}>
      {children}
    </WSContext.Provider>
  );
}

// -----------------------------------------------------------
// Hook
// -----------------------------------------------------------
export function useWebSocket() {
  const ctx = useContext(WSContext);
  if (!ctx) throw new Error("useWebSocket must be used within WebSocketProvider");
  return ctx;
}

/**
 * Subscribe to a specific WebSocket event with auto-cleanup.
 */
export function useWSEvent<T>(event: WSEvent, handler: (data: T) => void) {
  const { subscribe } = useWebSocket();

  useEffect(() => {
    const unsubscribe = subscribe(event, handler as (data: unknown) => void);
    return unsubscribe;
  }, [event, handler, subscribe]);
}
