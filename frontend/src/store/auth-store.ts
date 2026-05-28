import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";
import type { User, AuthSession } from "@/types";
import { apiClient } from "@/lib/api-client";

// -----------------------------------------------------------
// State Shape
// -----------------------------------------------------------
interface AuthState {
  user: User | null;
  accessToken: string | null;
  expiresAt: number | null;
  isLoading: boolean;
  error: string | null;
}

interface AuthActions {
  login: (email: string, password: string) => Promise<void>;
  register: (name: string, email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  refreshToken: () => Promise<void>;
  updateUser: (updates: Partial<User>) => void;
  clearError: () => void;
}

type AuthStore = AuthState & AuthActions;

// -----------------------------------------------------------
// Store
// -----------------------------------------------------------
export const useAuthStore = create<AuthStore>()(
  persist(
    immer((set, get) => ({
      // State
      user: null,
      accessToken: null,
      expiresAt: null,
      isLoading: false,
      error: null,

      // Actions
      login: async (email: string, password: string) => {
        set((state) => {
          state.isLoading = true;
          state.error = null;
        });

        try {
          const response = await apiClient.post<AuthSession>("/auth/login", {
            email,
            password,
          });

          set((state) => {
            state.user = response.data.user;
            state.accessToken = response.data.accessToken;
            state.expiresAt = response.data.expiresAt;
            state.isLoading = false;
          });
        } catch (err) {
          set((state) => {
            state.error = (err as Error).message || "Login failed";
            state.isLoading = false;
          });
          throw err;
        }
      },

      register: async (name: string, email: string, password: string) => {
        set((state) => {
          state.isLoading = true;
          state.error = null;
        });

        try {
          const response = await apiClient.post<AuthSession>("/auth/register", {
            name,
            email,
            password,
          });

          set((state) => {
            state.user = response.data.user;
            state.accessToken = response.data.accessToken;
            state.expiresAt = response.data.expiresAt;
            state.isLoading = false;
          });
        } catch (err) {
          set((state) => {
            state.error = (err as Error).message || "Registration failed";
            state.isLoading = false;
          });
          throw err;
        }
      },

      logout: async () => {
        const { accessToken } = get();
        if (accessToken) {
          try {
            await apiClient.post("/auth/logout");
          } catch {
            // Ignore logout errors — clear state anyway
          }
        }

        set((state) => {
          state.user = null;
          state.accessToken = null;
          state.expiresAt = null;
          state.error = null;
        });
      },

      refreshToken: async () => {
        try {
          const response = await apiClient.post<AuthSession>("/auth/refresh");
          set((state) => {
            state.accessToken = response.data.accessToken;
            state.expiresAt = response.data.expiresAt;
          });
        } catch {
          // Refresh failed — force logout
          set((state) => {
            state.user = null;
            state.accessToken = null;
            state.expiresAt = null;
          });
        }
      },

      updateUser: (updates: Partial<User>) => {
        set((state) => {
          if (state.user) {
            Object.assign(state.user, updates);
          }
        });
      },

      clearError: () => {
        set((state) => {
          state.error = null;
        });
      },
    })),
    {
      name: "neurosight-auth",
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        user: state.user,
        accessToken: state.accessToken,
        expiresAt: state.expiresAt,
      }),
    }
  )
);

// Computed selectors
export const selectIsAuthenticated = (state: AuthStore) =>
  !!state.user && !!state.accessToken && (state.expiresAt ?? 0) > Date.now();
