import axios, {
  type AxiosInstance,
  type AxiosRequestConfig,
  type InternalAxiosRequestConfig,
  type AxiosResponse,
  type AxiosError,
} from "axios";

// -----------------------------------------------------------
// Types
// -----------------------------------------------------------
interface QueuedRequest {
  resolve: (value: unknown) => void;
  reject: (error: unknown) => void;
}

// -----------------------------------------------------------
// Config
// -----------------------------------------------------------
const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const API_VERSION = "v1";

// -----------------------------------------------------------
// Create Instance
// -----------------------------------------------------------
const instance: AxiosInstance = axios.create({
  baseURL: `${BASE_URL}/api/${API_VERSION}`,
  timeout: 30000,
  headers: {
    "Content-Type": "application/json",
    Accept: "application/json",
  },
  withCredentials: true,
});

// -----------------------------------------------------------
// Token Refresh Logic
// -----------------------------------------------------------
let isRefreshing = false;
let failedQueue: QueuedRequest[] = [];

function processQueue(error: unknown, token: string | null = null) {
  failedQueue.forEach(({ resolve, reject }) => {
    if (error) {
      reject(error);
    } else {
      resolve(token);
    }
  });
  failedQueue = [];
}

// -----------------------------------------------------------
// Request Interceptor — Attach Bearer Token
// -----------------------------------------------------------
instance.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    // Dynamically read token from store on each request
    // This avoids stale closures
    const storeKey = "neurosight-auth";
    try {
      const stored = localStorage.getItem(storeKey);
      if (stored) {
        const { state } = JSON.parse(stored);
        if (state?.accessToken) {
          config.headers.Authorization = `Bearer ${state.accessToken}`;
        }
      }
    } catch {
      // localStorage unavailable (SSR or private browsing)
    }

    // Add request ID for tracing
    config.headers["X-Request-ID"] = crypto.randomUUID();

    return config;
  },
  (error) => Promise.reject(error)
);

// -----------------------------------------------------------
// Response Interceptor — Handle 401 & Token Refresh
// -----------------------------------------------------------
instance.interceptors.response.use(
  (response: AxiosResponse) => response,
  async (error: AxiosError) => {
    const originalRequest = error.config as InternalAxiosRequestConfig & {
      _retry?: boolean;
    };

    if (error.response?.status === 401 && !originalRequest._retry) {
      if (isRefreshing) {
        // Queue requests while refresh is in-flight
        return new Promise((resolve, reject) => {
          failedQueue.push({ resolve, reject });
        }).then((token) => {
          originalRequest.headers.Authorization = `Bearer ${token}`;
          return instance(originalRequest);
        });
      }

      originalRequest._retry = true;
      isRefreshing = true;

      try {
        const response = await instance.post<{ accessToken: string }>("/auth/refresh");
        const { accessToken } = response.data;

        // Update stored token
        const storeKey = "neurosight-auth";
        const stored = localStorage.getItem(storeKey);
        if (stored) {
          const parsed = JSON.parse(stored);
          parsed.state.accessToken = accessToken;
          localStorage.setItem(storeKey, JSON.stringify(parsed));
        }

        processQueue(null, accessToken);
        originalRequest.headers.Authorization = `Bearer ${accessToken}`;
        return instance(originalRequest);
      } catch (refreshError) {
        processQueue(refreshError, null);
        // Redirect to login
        window.location.href = "/auth/login";
        return Promise.reject(refreshError);
      } finally {
        isRefreshing = false;
      }
    }

    // Extract backend error message
    const serverMessage =
      (error.response?.data as { message?: string })?.message ||
      error.message ||
      "An unexpected error occurred";

    return Promise.reject(new Error(serverMessage));
  }
);

// -----------------------------------------------------------
// Typed API Client
// -----------------------------------------------------------
export const apiClient = {
  get: <T>(url: string, config?: AxiosRequestConfig) =>
    instance.get<T>(url, config).then((r) => r.data),

  post: <T>(url: string, data?: unknown, config?: AxiosRequestConfig) =>
    instance.post<T>(url, data, config).then((r) => r.data),

  put: <T>(url: string, data?: unknown, config?: AxiosRequestConfig) =>
    instance.put<T>(url, data, config).then((r) => r.data),

  patch: <T>(url: string, data?: unknown, config?: AxiosRequestConfig) =>
    instance.patch<T>(url, data, config).then((r) => r.data),

  delete: <T>(url: string, config?: AxiosRequestConfig) =>
    instance.delete<T>(url, config).then((r) => r.data),
};

export default instance;
