import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock axios before importing api-client
vi.mock("axios", () => {
  const mockInstance = {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    patch: vi.fn(),
    delete: vi.fn(),
    interceptors: {
      request: { use: vi.fn() },
      response: { use: vi.fn() },
    },
  };
  const axios = {
    default: { create: () => mockInstance },
    create: () => mockInstance,
  };
  return axios;
});

describe("API Client", () => {
  it("module imports without throwing", async () => {
    // Just verify the module loads cleanly
    const mod = await import("../api-client");
    expect(mod).toBeDefined();
    expect(mod.apiClient).toBeDefined();
  });

  it("apiClient exposes get, post, put, patch, delete", async () => {
    const { apiClient } = await import("../api-client");
    expect(typeof apiClient.get).toBe("function");
    expect(typeof apiClient.post).toBe("function");
    expect(typeof apiClient.put).toBe("function");
    expect(typeof apiClient.patch).toBe("function");
    expect(typeof apiClient.delete).toBe("function");
  });
});
